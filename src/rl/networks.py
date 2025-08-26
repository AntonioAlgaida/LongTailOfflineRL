# src/rl/networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# --- 1. Reusable State Encoder Module ---

class StateEncoder(nn.Module):
    """
    (V3 - Optimized) A reusable module that processes a structured dictionary of
    state tensors and produces a context-aware representation. This is the shared
    "perception backbone" for both the Actor and Critic.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        cql_cfg = config['cql']
        features_cfg = config['features']
        
        self.embed_dim = cql_cfg.get('embed_dim', 96)
        
        # --- Define Raw Feature Dimensions ---
        ego_dim = 1
        agent_dim = 10
        map_dim = features_cfg['map_points_per_polyline'] * 2
        tl_dim = 2
        goal_dim = 2

        # --- Use simpler, single-layer encoders for efficiency ---
        self.ego_encoder = nn.Linear(ego_dim, self.embed_dim)
        self.agent_encoder = nn.Linear(agent_dim, self.embed_dim)
        self.map_encoder = nn.Linear(map_dim, self.embed_dim)
        self.tl_encoder = nn.Linear(tl_dim, self.embed_dim)
        self.goal_encoder = nn.Linear(goal_dim, self.embed_dim)
        
        # --- Attention Module ---
        num_heads = cql_cfg.get('num_attention_heads', 4)
        self.cross_attention = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)
        self.attention_ln = nn.LayerNorm(self.embed_dim)

        # --- Define the final output dimension for downstream modules ---
        self.output_dim = self.embed_dim * 4 # ego + attention + goal_summary + traffic_light

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        ego_embedding = F.relu(self.ego_encoder(state['ego']))
        agent_embeddings = F.relu(self.agent_encoder(state['agents']))
        map_embeddings = F.relu(self.map_encoder(state['map']))
        tl_embedding = F.relu(self.tl_encoder(state['traffic_lights']))
        goal_embeddings = F.relu(self.goal_encoder(state['goal']))

        goal_summary_embedding = goal_embeddings.mean(dim=1)
        query_embedding = ego_embedding + goal_summary_embedding
        
        query = query_embedding.unsqueeze(1)
        context = torch.cat([agent_embeddings, map_embeddings], dim=1)
        padding_mask = torch.cat([state['agents_mask'], state['map_mask']], dim=1)
        
        attention_output, _ = self.cross_attention(
            query=query, key=context, value=context,
            key_padding_mask=~padding_mask.bool()
        )
        attention_output = self.attention_ln(attention_output.squeeze(1))
        
        return torch.cat([
            ego_embedding, 
            attention_output, 
            goal_summary_embedding,
            tl_embedding
        ], dim=1)
    

class Actor(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        self.state_encoder = StateEncoder(config)
        
        cql_cfg = config['cql']
        policy_head_input_dim = self.state_encoder.output_dim
        hidden_dim = cql_cfg['hidden_layers'][0]
        
        self.policy_head = nn.Sequential(
            nn.Linear(policy_head_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh()
        )
        self._initialize_action_scaling()

    def _initialize_action_scaling(self):
        action_cfg = self.cfg['action_space']
        min_accel = action_cfg['min_acceleration']
        max_accel = action_cfg['max_acceleration']
        max_yaw_rate = action_cfg['max_yaw_rate']
        scale = torch.tensor([(max_accel - min_accel) / 2.0, (max_yaw_rate * 2) / 2.0])
        bias = torch.tensor([(max_accel + min_accel) / 2.0, 0.0])
        self.register_buffer('action_scale', scale)
        self.register_buffer('action_bias', bias)

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        state_representation = self.state_encoder(state)
        squashed_action = self.policy_head(state_representation)
        final_action = squashed_action * self.action_scale + self.action_bias
        return final_action


# --- 3. The Critic Network (Standard Double Critic) ---
class DoubleCritic(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        # Use a single, shared state encoder for both Q-functions to save VRAM
        self.state_encoder = StateEncoder(config)
        self.q1_head = self._create_q_head()
        self.q2_head = self._create_q_head()

    def _create_q_head(self) -> nn.Module:
        cql_cfg = self.cfg['cql']
        action_dim = self.cfg['action_space']['dim']
        q_head_input_dim = self.state_encoder.output_dim + action_dim
        hidden_dim = cql_cfg['hidden_layers'][0]
        dropout_rate = self.cql_cfg.get('dropout_rate', 0.1) # Add to config

        return nn.Sequential(
            nn.Linear(q_head_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_representation = self.state_encoder(state)
        q_input = torch.cat([state_representation, action], dim=1)
        q1 = self.q1_head(q_input)
        q2 = self.q2_head(q_input)
        return q1, q2