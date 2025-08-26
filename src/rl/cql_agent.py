# src/rl/cql_agent.py

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import numpy as np
from typing import Dict, Tuple
import copy
import os

from src.rl.networks import Actor, DoubleCritic
from src.rl.reward_function import compute_reward_v3
from src.utils.loss import WeightedMSELoss # <-- ADD THIS IMPORT

class CQLAgent:
    """
    (V2 - Structured) An implementation of Conservative Q-Learning that operates
    on structured, dictionary-based state representations.
    """
    def __init__(self, config: Dict, device: torch.device):
        self.cfg = config
        self.cql_cfg = config['cql']
        self.device = device

        # --- 1. Load and store structured normalization statistics ---
        print("CQLAgent: Loading structured normalization statistics...")
        stats_path = config['data']['feature_stats_path_v2'] # Use the V2 stats
        stats = np.load(stats_path)
        self.state_mean = {k.replace('_mean',''): torch.from_numpy(v).to(device).float() for k,v in stats.items() if '_mean' in k}
        self.state_std = {k.replace('_std',''): torch.from_numpy(v).to(device).float() for k,v in stats.items() if '_std' in k}
        
        # --- 2. Initialize Networks (they now take dicts) ---
        self.actor = Actor(config).to(device)
        self.critic = DoubleCritic(config).to(device)
        
        # if torch.__version__ >= "2.0.0":
        #     print("Compiling models with torch.compile...")
        #     self.actor = torch.compile(self.actor, mode="reduce-overhead")
        #     self.critic = torch.compile(self.critic, mode="max-autotune")
        #     print("Compilation complete.")
            
        self.critic_target = copy.deepcopy(self.critic)
        
        # --- 2. Initialize Target Networks ---
        # Target networks are time-delayed copies of the main networks that are
        # updated slowly. This provides a stable target for the Bellman backup.
        self.critic_target = copy.deepcopy(self.critic)
        
        # --- 3. Optimizers and Hyperparameters (no change) ---
        lr_actor = float(self.cql_cfg['learning_rate_actor'])
        lr_critic = float(self.cql_cfg['learning_rate_critic'])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = self.cql_cfg['gamma']
        self.tau = self.cql_cfg['tau']
        self.cql_alpha = self.cql_cfg['cql_alpha']
        self.cql_n_actions = self.cql_cfg.get('cql_n_actions', 10)
        
        self.bc_alpha_initial = self.cql_cfg.get('bc_alpha_initial', 0.1)
        self.bc_alpha_final = self.cql_cfg.get('bc_alpha_final', 1.0)
        self.bc_alpha_decay_steps = self.cql_cfg.get('bc_alpha_decay_steps', 500000)
        
        # Initialize a gradient scaler for mixed precision training
        self.scaler = GradScaler(enabled=(device.type == 'cuda'))
        
        # --- NEW: Instantiate the WeightedMSELoss for the BC component ---
        print("CQLAgent: Initializing WeightedMSELoss...")
        weights_path = os.path.join(
            os.path.dirname(config['data']['feature_stats_path_v2']), 
            'action_weights_v2.pt'
        )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Action weights not found at: {weights_path}")
            
        self.bc_loss_fn = WeightedMSELoss(weights_path).to(device)

    def _normalize_states(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Helper to apply normalization to each tensor in the state dictionary."""
        norm_dict = {}
        for key, tensor in state_dict.items():
            if key.endswith('_mask'):
                norm_dict[key] = tensor # Don't normalize masks
            else:
                norm_tensor = (tensor - self.state_mean[key]) / (self.state_std[key] + 1e-6)
                norm_dict[key] = torch.clamp(norm_tensor, -10.0, 10.0)
        return norm_dict

    def update(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """
        Performs a single CQL update step on a batch of data.
        
        Args:
            batch: A dictionary containing 'states', 'actions', 'rewards', 
                   'next_states', and 'dones' tensors.
            step: The current global training step, used for scheduling.

        Returns:
            A dictionary of loss values for logging.
        """
        states_raw, actions, next_states_raw, dones = self._unpack_batch(batch)
        
        # The reward function needs the structured state dictionary
        rewards, reward_components = compute_reward_v3(
            states_raw, actions, next_states_raw, dones, self.cfg
        )

        states = self._normalize_states(states_raw)
        next_states = self._normalize_states(next_states_raw)
        
        # --- 1. Critic Update (The Core of CQL) ---
        
        # Get current Q-values for the (state, action) pairs in the batch
        with autocast(device_type=self.device.type, dtype=torch.float16):
            q1_current, q2_current = self.critic(states, actions)
            
            # --- Standard Bellman backup for the target ---
            with torch.no_grad():
                next_actions = self.actor(next_states)
                q1_next_target, q2_next_target = self.critic_target(next_states, next_actions)
                q_next_target = torch.min(q1_next_target, q2_next_target)
                q_target = rewards + (1.0 - dones.unsqueeze(-1)) * self.gamma * q_next_target
                    
            # --- Bellman Error Loss ---
            critic_loss_bellman = F.mse_loss(q1_current, q_target) + F.mse_loss(q2_current, q_target)
            
            # --- CQL Conservative Regularizer ---
            # This is the term that makes CQL conservative. It pushes down the Q-values
            # of out-of-distribution actions while pushing up the Q-values of actions
            # from the dataset.
            
            # a) Get Q-values for actions sampled from the CURRENT policy
            q1_policy_actions, q2_policy_actions = self._get_policy_action_q_values(states)
            
            # b) Get Q-values for actions sampled randomly
            q1_random_actions, q2_random_actions = self._get_random_action_q_values(states)

            # c) Concatenate all sampled Q-values and the Q-values of the batch actions
            #    log(sum(exp(Q))) is a smooth approximation of max(Q)
            q1_current = q1_current.unsqueeze(1)
            q2_current = q2_current.unsqueeze(1)
            q1_cat = torch.cat([q1_policy_actions, q1_random_actions, q1_current], dim=1)
            q2_cat = torch.cat([q2_policy_actions, q2_random_actions, q2_current], dim=1)
            
            cql_logsumexp1 = torch.logsumexp(q1_cat, dim=1).mean()
            cql_logsumexp2 = torch.logsumexp(q2_cat, dim=1).mean()
            
            # The conservative loss is the logsumexp term minus the Q-values of the dataset actions
            cql_loss_q1 = cql_logsumexp1 - q1_current.mean()
            cql_loss_q2 = cql_logsumexp2 - q2_current.mean()
            critic_loss_cql = self.cql_alpha * (cql_loss_q1 + cql_loss_q2)

            # --- Final Critic Loss and Update ---
            critic_loss = critic_loss_bellman + critic_loss_cql
        
        self.critic_optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for a small speedup
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.scaler.step(self.critic_optimizer)

        # --- 2. Actor Update (MODIFIED with BC Auxiliary Loss) ---
        # The actor is updated to maximize the Q-value of its chosen actions.
        # We detach the critic here so gradients only flow through the actor.
        
        # First, calculate the current alpha value based on the decay schedule
        progress = min(step / self.bc_alpha_decay_steps, 1.0)
        current_alpha = self.bc_alpha_initial + (self.bc_alpha_final - self.bc_alpha_initial) * progress
        
        # Freeze critic parameters
        for p in self.critic.parameters():
            p.requires_grad = False
        
        # --- RL Component of the loss ---
        with autocast(device_type=self.device.type, dtype=torch.float16):
            actor_actions = self.actor(states)
            q1_actor, q2_actor = self.critic(states, actor_actions)
            q_actor = torch.min(q1_actor, q2_actor)
            actor_loss_rl = -q_actor.mean()
            
            # --- BC Component of the loss ---
            # `actions` in the batch are the expert actions
            # actor_actions_bc = self.actor(states)
            
            # min_accel = self.cfg['action_space']['min_acceleration']
            # max_accel = self.cfg['action_space']['max_acceleration']
            # max_yaw = self.cfg['action_space']['max_yaw_rate']
            
            # clamped_expert_actions = torch.clone(actions)
            # clamped_expert_actions[:, 0] = torch.clamp(actions[:, 0], min_accel, max_accel)
            # clamped_expert_actions[:, 1] = torch.clamp(actions[:, 1], -max_yaw, max_yaw)
            # --- END OF FIX ---
            
            actor_loss_bc = 0#self.bc_loss_fn(actor_actions, clamped_expert_actions)
            
            # --- Combine the losses ---
            # actor_loss = (current_alpha * actor_loss_rl) + ((1 - current_alpha) * actor_loss_bc)
            # Disable the bc_actor_loss
            actor_loss = actor_loss_rl
        
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.scaler.step(self.actor_optimizer)
        
        # --- Update the scaler for the next iteration ---
        self.scaler.update()

        # Unfreeze critic parameters
        for p in self.critic.parameters():
            p.requires_grad = True

        # --- 3. Target Network Update ---
        self.soft_update_target_network()
        
        # --- 4. NEW: Gather all metrics for logging ---
        with torch.no_grad():
            metrics = {
                # === Reward Metric ===
                'Reward/batch_reward_mean': rewards.mean().item(),
                'Reward/batch_reward_std': rewards.std().item(),
                'Reward/batch_reward_min': rewards.min().item(),
                'Reward/batch_reward_max': rewards.max().item(),
                
                # Core Losses
                'Loss/critic_total': critic_loss.item(),
                'Loss/critic_bellman': critic_loss_bellman.item(),
                'Loss/critic_cql': critic_loss_cql.item(),
                'Loss/actor': actor_loss.item(),
                'Loss/actor_rl': actor_loss_rl.item(),
                'Loss/actor_bc': 0,
                'bc_alpha': current_alpha,
                
                # Q-Value Statistics
                'Q_Values/dataset_actions': q1_current.mean().item(),
                'Q_Values/policy_actions': q1_policy_actions.mean().item(),
                'Q_Values/random_actions': q1_random_actions.mean().item(),
                'Q_Values/target': q_target.mean().item(),

                # Actor Action Statistics
                'Actor/pred_accel_mean': actor_actions[:, 0].mean().item(),
                'Actor/pred_accel_std': actor_actions[:, 0].std().item(),
                'Actor/pred_yaw_rate_mean': actor_actions[:, 1].mean().item(),
                'Actor/pred_yaw_rate_std': actor_actions[:, 1].std().item(),
            }
            
            # --- NEW GENERIC LOGGING LOOP ---
            # Add the mean of each individual reward component to the metrics dict
            for key, tensor in reward_components.items():
                metrics[f'Reward/Components/{key}'] = tensor.mean().item()
        
        return metrics

    def _get_policy_action_q_values(self, states: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (V2 - Structured) Samples actions from the current policy and evaluates their Q-values.
        """
        batch_size = states['ego'].shape[0]
        
        # --- CORRECTED: Repeat the structured state dictionary N times ---
        repeated_states = {}
        for key, tensor in states.items():
            # tensor shape: (B, ...) -> (B, 1, ...) -> (B, N, ...) -> (B*N, ...)
            # We use `*tensor.shape[1:]` to handle tensors with different numbers of dimensions.
            repeated_tensor = tensor.unsqueeze(1).repeat(1, self.cql_n_actions, *([1]*(tensor.dim()-1)))
            repeated_states[key] = repeated_tensor.view(batch_size * self.cql_n_actions, *tensor.shape[1:])
        
        # Now, `repeated_states` is a valid state dictionary for a larger batch
        sampled_actions = self.actor(repeated_states)
        q1, q2 = self.critic(repeated_states, sampled_actions)
        
        # Reshape the Q-values back to (batch_size, n_actions, 1) for the logsumexp
        return q1.view(batch_size, self.cql_n_actions, 1), q2.view(batch_size, self.cql_n_actions, 1)
        
    def _get_random_action_q_values(self, states: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (V2 - Structured) Samples actions uniformly and evaluates their Q-values.
        """
        batch_size = states['ego'].shape[0]

        # --- CORRECTED: The state repetition logic is identical to the function above ---
        repeated_states = {}
        for key, tensor in states.items():
            repeated_tensor = tensor.unsqueeze(1).repeat(1, self.cql_n_actions, *([1]*(tensor.dim()-1)))
            repeated_states[key] = repeated_tensor.view(batch_size * self.cql_n_actions, *tensor.shape[1:])

        # Sample random actions in the normalized [-1, 1] range
        # The total number of actions needed is batch_size * cql_n_actions
        total_actions = batch_size * self.cql_n_actions
        random_actions_squashed = torch.rand(
            total_actions, 
            self.cfg['action_space']['dim'], 
            device=self.device
        ) * 2.0 - 1.0 # Correct uniform sampling from -1 to 1
        
        # Rescale them to the physical space, using the actor's scaling buffers
        random_actions = random_actions_squashed * self.actor.action_scale + self.actor.action_bias
        
        q1, q2 = self.critic(repeated_states, random_actions)
        
        return q1.view(batch_size, self.cql_n_actions, 1), q2.view(batch_size, self.cql_n_actions, 1)

    def soft_update_target_network(self):
        """Performs a soft update of the target critic network."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def _unpack_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple:
        """Helper to move all tensors in the batch to the correct device."""
        # The batch is now a dictionary of dictionaries
        states = {k: v.to(self.device) for k, v in batch['states'].items()}
        actions = batch['actions'].to(self.device)
        next_states = {k: v.to(self.device) for k, v in batch['next_states'].items()}
        dones = batch['dones'].to(self.device)
        return states, actions, next_states, dones
    
    def save(self, directory: str, filename: str):
        """Saves the actor and critic models."""
        torch.save(self.actor.state_dict(), os.path.join(directory, f"{filename}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{filename}_critic.pth"))

    def load(self, directory: str, filename: str):
        """Loads the actor and critic models."""
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor.pth"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic.pth"), map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)