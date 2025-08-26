# src/rl/networks.py

import torch
import torch.nn as nn
from typing import Dict, Tuple

# --- 1. State Reconstruction Utility ---

class StateReconstructor:
    """
    A utility class to efficiently reconstruct the structured state dictionary
    from a flattened state vector on-the-fly during training. This decouples
    the on-disk data format from the model's architecture.
    """
    def __init__(self, config: Dict):
        # Pre-calculate all shapes and slice indices from the config
        features_cfg = config['features']
        
        # --- Component Dimensions ---
        self.ego_dim = 1
        self.agent_feature_dim = 7   # [x, y, vx, vy, cos(h), sin(h), valid]
        self.map_feature_dim = (features_cfg['map_points_per_polyline'] * 2) + 1
        self.tl_dim = 2
        
        self.n_agents = features_cfg['num_agents']
        self.n_map = features_cfg['num_map_polylines']
        
        # --- Flattened Block Sizes ---
        self.agent_block_size = self.n_agents * self.agent_feature_dim
        self.map_block_size = self.n_map * self.map_feature_dim
        
        # --- Slice Indices for the flat vector ---
        self.start_ego, self.end_ego = 0, self.ego_dim
        self.start_agent, self.end_agent = self.end_ego, self.end_ego + self.agent_block_size
        self.start_map, self.end_map = self.end_agent, self.end_agent + self.map_block_size
        self.start_tl, self.end_tl = self.end_map, self.end_map + self.tl_dim

    def __call__(self, flat_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Reconstructs the state dictionary from a batch of flat state vectors.

        Args:
            flat_states: A tensor of shape (B, flattened_feature_dim).

        Returns:
            A dictionary of structured tensors ready for the Actor model.
        """
        batch_size = flat_states.shape[0]
        
        # --- Extract and Reshape Each Component using pre-calculated indices ---
        ego = flat_states[:, self.start_ego:self.end_ego]
        
        agents_flat = flat_states[:, self.start_agent:self.end_agent]
        agents = agents_flat.view(batch_size, self.n_agents, self.agent_feature_dim)
        
        map_flat = flat_states[:, self.start_map:self.end_map]
        map_lanes = map_flat.view(batch_size, self.n_map, self.map_feature_dim)
        
        traffic_lights = flat_states[:, self.start_tl:self.end_tl]
        
        # Extract validity masks from the last column of the feature vectors
        agents_mask = agents[:, :, -1].bool()
        map_mask = map_lanes[:, :, -1].bool()
        
        return {
            'ego': ego,
            'agents': agents,
            'agents_mask': agents_mask,
            'map': map_lanes,
            'map_mask': map_mask,
            'traffic_lights': traffic_lights
        }

# --- 2. The Actor Network (Entity Embeddings + Cross-Attention) ---

class Actor(nn.Module):
    """
    An attention-based Actor that ingests a flat state vector, reconstructs it
    into a structured dictionary of entities, and then uses cross-attention
    to produce a context-aware action.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        
        # Instantiate the reconstructor utility
        self.reconstructor = StateReconstructor(config)
        
        cql_cfg = config['cql']
        embed_dim = cql_cfg.get('embed_dim', 128)
        
        # Get feature dimensions from the reconstructor for clarity
        ego_dim = self.reconstructor.ego_dim
        agent_dim = self.reconstructor.agent_feature_dim
        map_dim = self.reconstructor.map_feature_dim
        tl_dim = self.reconstructor.tl_dim

        # --- Individual Entity Encoders ---
        self.ego_encoder = self._create_encoder(ego_dim, embed_dim)
        self.agent_encoder = self._create_encoder(agent_dim, embed_dim)
        self.map_encoder = self._create_encoder(map_dim, embed_dim)
        self.tl_encoder = self._create_encoder(tl_dim, embed_dim)
        
        # --- Cross-Attention Module ---
        num_heads = cql_cfg.get('num_attention_heads', 4)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        self.attention_ln = nn.LayerNorm(embed_dim)

        # --- Final Policy Head ---
        policy_head_input_dim = embed_dim * 2 # (ego_embedding + attention_output)
        hidden_dim = cql_cfg['hidden_layers'][0]
        
        self.policy_head = nn.Sequential(
            nn.Linear(policy_head_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2), # 2 action dimensions
            nn.Tanh()                      
        )
        
        self._initialize_action_scaling()

    def _create_encoder(self, input_dim: int, output_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def _initialize_action_scaling(self):
        action_cfg = self.cfg['action_space']
        min_accel = action_cfg['min_acceleration']
        max_accel = action_cfg['max_acceleration']
        max_yaw_rate = action_cfg['max_yaw_rate']
        scale = torch.tensor([(max_accel - min_accel) / 2.0, (max_yaw_rate * 2) / 2.0])
        bias = torch.tensor([(max_accel + min_accel) / 2.0, 0.0])
        self.register_buffer('action_scale', scale)
        self.register_buffer('action_bias', bias)

    def forward(self, flat_state: torch.Tensor) -> torch.Tensor:
        # --- Step 1: Reconstruct the structured state dictionary ---
        state_dict = self.reconstructor(flat_state)
        
        # --- Step 2: Encode all entities ---
        ego_embedding = self.ego_encoder(state_dict['ego'])
        agent_embeddings = self.agent_encoder(state_dict['agents'])
        map_embeddings = self.map_encoder(state_dict['map'])
        tl_embedding = self.tl_encoder(state_dict['traffic_lights'])
        
        # --- Step 3: Prepare for Attention ---
        query = ego_embedding.unsqueeze(1)
        context = torch.cat([agent_embeddings, map_embeddings, tl_embedding.unsqueeze(1)], dim=1)
        
        # The traffic light is always a single, real entity
        tl_mask = torch.ones(state_dict['traffic_lights'].shape[0], 1, device=flat_state.device).bool()
        padding_mask = torch.cat([state_dict['agents_mask'], state_dict['map_mask'], tl_mask], dim=1)
        
        # --- Step 4: Apply Cross-Attention ---
        attention_output, _ = self.cross_attention(
            query=query, key=context, value=context,
            key_padding_mask=~padding_mask
        )
        attention_output = self.attention_ln(attention_output.squeeze(1))
        
        # --- Step 5: Final Decision ---
        combined_features = torch.cat([ego_embedding, attention_output], dim=1)
        squashed_action = self.policy_head(combined_features)
        final_action = squashed_action * self.action_scale + self.action_bias
        
        return final_action

# --- 3. The Critic Network (Standard Double Critic) ---

class DoubleCritic(nn.Module):
    """
    A standard Double Critic network that operates on a flattened state vector
    and an action tensor.
    """
    def __init__(self, config: Dict):
        super().__init__()
        
        # --- Determine the total flattened state dimension from config ---
        features_cfg = config['features']
        ego_dim = 1
        agent_dim = features_cfg['num_agents'] * 7
        map_dim = features_cfg['num_map_polylines'] * (features_cfg['map_points_per_polyline'] * 2 + 1)
        tl_dim = 2
        state_dim = ego_dim + agent_dim + map_dim + tl_dim
        
        action_dim = config['action_space']['dim']
        input_dim = state_dim + action_dim
        hidden_dim = config['cql']['hidden_layers'][0]

        # Define Q1 and Q2 networks
        self.q1 = self._create_q_network(input_dim, hidden_dim)
        self.q2 = self._create_q_network(input_dim, hidden_dim)
        
    def _create_q_network(self, input_dim: int, hidden_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, flat_state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate the already flat state with the action
        sa = torch.cat([flat_state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2