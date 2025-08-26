# src/rl/train.py

# The main training script for all CQL agents.
# To run an experiment:
# conda activate wwm
# python -m src.rl.train --agent_type heuristic --run_name CQL_H_run1_noBC
# CUDA_LAUNCH_BLOCKING=1 python -m src.rl.train --agent_type heuristic --run_name CQL_H_run1_noBC
# python -m src.rl.train --agent_type heuristic_scenario --run_name CQL_HS_scenario_run1_noBC
# python -m src.rl.train --agent_type ensemble --run_name CQL_E_run1_noBC
# python -m src.rl.train --agent_type ensemble_scenario --run_name CQL_ES_scenario_run1_noBC
# python -m src.rl.train --agent_type action_rarity --run_name CQL_AR_run1_noBC
# python -m src.rl.train --agent_type action_rarity_scenario --run_name CQL_ARS_scenario_run1_noBC
# baseline agent:
# python -m src.rl.train --agent_type baseline --run_name CQL_B_run1

import os
# This tells JAX to not pre-allocate GPU memory.
# It's a common fix for JAX/PyTorch interoperability.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# This tells TensorFlow to use the CPU only. It prevents TF from
# initializing on the GPU and causing the cuDNN factory registration warnings.

import sys
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from glob import glob
from tqdm import tqdm
import numpy as np
import datetime
import argparse
import multiprocessing as mp
from torch.nn import functional as F
import traceback
# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.rl.dataset import OfflineRLTimestepDataset, OfflineRLScenarioDataset
from src.rl.cql_agent import CQLAgent
from torch.utils.tensorboard import SummaryWriter
from src.rl.reward_function import compute_reward_v3
from src.utils.collations import structured_collate_fn # We need a collate_fn now

# --- NEW: Evaluation Function ---
@torch.no_grad()
def evaluate_agent(agent: CQLAgent, val_loader: DataLoader, num_eval_batches: int, config: dict):
    agent.actor.eval()
    agent.critic.eval()
    
    total_q_value, total_mse, total_reward = 0.0, 0.0, 0.0
    batches_processed = 0
    total_reward_components = {}

    for i, batch in enumerate(val_loader):
        if i >= num_eval_batches: break
        
        # --- CORRECTED: Unpack the structured batch ---
        states_raw, expert_actions, next_states_raw, dones = agent._unpack_batch(batch)

        # Normalize states for the models
        states = agent._normalize_states(states_raw)
        
        policy_actions = agent.actor(states)
        q1, q2 = agent.critic(states, policy_actions)
        q_value = torch.min(q1, q2).mean().item()
        total_q_value += q_value
        
        # We need the config for the reward function
        calculated_rewards, reward_components = compute_reward_v3(
            states_raw, policy_actions, next_states_raw, dones, config
        )
        total_reward += calculated_rewards.mean().item()
        
        for key, tensor in reward_components.items():
            if key not in total_reward_components:
                total_reward_components[key] = 0.0
            total_reward_components[key] += tensor.mean().item()
        
        mse = F.mse_loss(policy_actions, expert_actions).item()
        total_mse += mse
        batches_processed += 1

    # Avoid division by zero if val_loader is empty
    if batches_processed == 0: return {}

    avg_q_value = total_q_value / batches_processed
    avg_mse = total_mse / batches_processed
    avg_reward = total_reward / batches_processed
    
    agent.actor.train()
    agent.critic.train()
    
    # --- Final metrics dictionary ---
    eval_metrics = {
        'Evaluation/avg_reward': avg_reward,
        'Evaluation/avg_q_value': avg_q_value,
        'Evaluation/mse_vs_expert': avg_mse
    }
    
    # --- NEW: Add the averaged components to the final dict ---
    for key, total_value in total_reward_components.items():
        eval_metrics[f'Evaluation/Components/{key}'] = total_value / batches_processed
    
    return eval_metrics
    

def main():
    # --- 1. Argument Parsing and Configuration ---
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method already set.")
        
    parser = argparse.ArgumentParser(description="Train a CQL agent with a specified data sampling strategy.")
    parser.add_argument('--agent_type', type=str, required=False, default='heuristic_scenario',
                        choices=['baseline', 'heuristic', 'ensemble', 'action_rarity', 
                                 'heuristic_scenario', 'ensemble_scenario', 'action_rarity_scenario'],
                        help="The type of data weighting strategy to use.")
    parser.add_argument('--run_name', type=str, required=False, default='debug_run',
                        help="A unique name for this training run for logging purposes.")
    args = parser.parse_args()
    
    config = load_config()
    cql_cfg = config['cql']
    data_cfg = config['data']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Training for Agent: {args.agent_type} on device: {device} ---")

    # --- 2. Setup Logging and Checkpointing ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f"{args.run_name}_{timestamp}")
    model_dir = os.path.join('models', f"{args.run_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Model checkpoints will be saved to: {model_dir}")

    # --- 3. Data Pipeline Setup (The Core Logic) ---
    train_files = sorted(glob(os.path.join(data_cfg['featurized_dir_v2'], 'training', '*.pt')))
    val_files = sorted(glob(os.path.join(data_cfg['featurized_dir_v2'], 'validation', '*.pt')))

    # Determine if we are using timestep or scenario weighting
    is_scenario_level = '_scenario' in args.agent_type
    
    sampler = None
    train_dataset = None
    
    if is_scenario_level:
        # --- SCENARIO-LEVEL WEIGHTING ---
        print(f"Using scenario-level weighting for agent type: {args.agent_type}")
        agent_name = args.agent_type.replace('_scenario', '')
        scores_path = os.path.join(data_cfg['criticality_scores_dir_v2'], 'scenario_level', f'{agent_name}_scenario_scores.pt')
        
        if not os.path.exists(scores_path):
            raise FileNotFoundError(f"Scenario score file not found: {scores_path}")
        
        print(f"Loading scenario scores from: {scores_path}")
        scenario_scores = torch.load(scores_path, weights_only=False)
        train_dataset = OfflineRLScenarioDataset(train_files, scenario_scores)
        
        # IterableDataset doesn't use a sampler, shuffle must be False
        train_loader = DataLoader(
            train_dataset,
            batch_size=cql_cfg['batch_size'],
            shuffle=False,
            num_workers=data_cfg['num_workers'],
            pin_memory=True,
            persistent_workers=True if data_cfg['num_workers'] > 0 else False,
            collate_fn=structured_collate_fn # Add the collate_fn
        )
        
    else:
        # --- TIMESTEP-LEVEL WEIGHTING (or UNIFORM for baseline) ---
        print(f"Using timestep-level weighting for agent type: {args.agent_type}")
        train_dataset = OfflineRLTimestepDataset(train_files)
        sampler = None # Default for baseline
        
        if args.agent_type != 'baseline':
            score_dir = os.path.join(data_cfg['criticality_scores_dir_v2'], 'timestep_level', args.agent_type)
            
            # This is a slow operation, let's build a dedicated function
            # For now, let's assume a function `load_and_align_weights` exists
            weights = train_dataset.load_and_align_weights(score_dir, config) # This method needs to be in the Dataset class
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cql_cfg['batch_size'],
            sampler=sampler,
            shuffle=(sampler is None and not is_scenario_level), # Shuffle only if no sampler is used (i.e., for baseline)
            num_workers=data_cfg['num_workers'],
            pin_memory=True,
            persistent_workers=True if data_cfg['num_workers'] > 0 else False,
            collate_fn=structured_collate_fn # Add the collate_fn
        )
    
    # --- Validation Dataset and DataLoader Setup ---
    # The validation dataset is always a simple, unweighted TimestepDataset
    val_dataset = OfflineRLTimestepDataset(val_files)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cql_cfg['batch_size'],
        shuffle=False, 
        num_workers=0, # <-- SET TO 0
        pin_memory=False, # <-- SET TO False, not needed for num_workers=0
        collate_fn=structured_collate_fn # Add the collate_fn
    )

    # --- 4. Initialize Agent ---
    agent = CQLAgent(config, device)

    # --- 5. Main Training Loop ---
    total_steps = cql_cfg['num_train_steps']
    eval_interval = cql_cfg.get('eval_interval_steps', 10000) # Get from config
    num_eval_batches = cql_cfg.get('num_eval_batches', 50)
    
    train_iterator = iter(train_loader)
    
    for step in tqdm(range(total_steps), desc=f"Training {args.run_name}"):
        try:
            batch = next(train_iterator)
        except StopIteration:
            # print(f'Error: {traceback.format_exc()}')
            # DataLoader is exhausted, restart it for the next epoch/pass
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            
        # Perform a single update step
        try:
            losses = agent.update(batch, step)
        except RuntimeError as e:
            print(f"!!! CRASH DETECTED at step {step} !!!")
            print(f"Saving problematic batch to 'debug_batch.pt'")
            torch.save(batch, 'debug_batch.pt')
            raise e
        # Log losses to TensorBoard
        if (step + 1) % 100 == 0: # Log every 100 steps
            for key, value in losses.items():
                writer.add_scalar(key, value, step)
        
        # --- NEW: Periodic Evaluation Block ---
        if (step + 1) % eval_interval == 0:
            print(f"\nStep {step+1}: Running evaluation...")
            eval_metrics = evaluate_agent(agent, val_loader, num_eval_batches, config)
            
            for key, value in eval_metrics.items():
                writer.add_scalar(key, value, step)
            
            print(f"  -> Eval Metrics: {eval_metrics}")
            
            # Optionally, save a checkpoint after each evaluation
            print(f"\nStep {step+1}: Saving model checkpoint...")
            agent.save(model_dir, f'step_{step+1}')
            print(f"  -> Model checkpoint saved.")

    # --- Final Save ---
    agent.save(model_dir, 'final')
    writer.close()
    print(f"\n--- Training Complete for {args.run_name} ---")
    print(f"Final models saved in {model_dir}")


if __name__ == '__main__':
    main()