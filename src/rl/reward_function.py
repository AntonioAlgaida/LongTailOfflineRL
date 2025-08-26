# src/rl/reward_function.py

import torch
from typing import Dict, Tuple

def compute_reward_v1(
    state: torch.Tensor, 
    action: torch.Tensor, 
    next_state: torch.Tensor, 
    done: torch.Tensor
    ) -> torch.Tensor:
    """
    Computes a reward based on the SDC's forward speed.
    
    This requires reconstructing the ego state to get the SDC's speed.
    For now, let's keep it simple and use a placeholder until the full
    state reconstruction is available in the training loop.
    
    Let's refine this. The 'ego' feature in our simplified model is speed.
    The state vector passed to this function will be the FLAT vector.
    
    Args:
        state: The flat state tensor for the current timestep, shape (D,).
        action: The action tensor, shape (2,).
        next_state: The flat state tensor for the next timestep, shape (D,).
        done: The done tensor, shape (1,).

    Returns:
        A scalar reward tensor.
    """
    # The first feature in our simplified flat vector is the SDC's speed.
    sdc_speed = state[0]
    
    # We can normalize the speed to a reasonable range.
    # Let's say a max speed of 20 m/s (~45 mph) gets the max reward.
    max_speed_for_reward = 20.0
    
    # Reward is linearly proportional to speed, clipped at the max.
    progress_reward = torch.clamp(sdc_speed / max_speed_for_reward, 0, 1)
    
    # We can add a small penalty for extreme actions to encourage smoothness.
    comfort_penalty = 0.01 * (action[0].abs() > 5.0) + 0.01 * (action[1].abs() > 0.8)
    
    final_reward = progress_reward - comfort_penalty
    
    return final_reward.unsqueeze(0)

def compute_reward_v2(
    state: torch.Tensor, 
    action: torch.Tensor, 
    next_state: torch.Tensor, 
    done: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    (V2 - Sophisticated) A multi-objective reward function for the CQL agent.
    
    This function computes a scalar reward as a weighted sum of components related to:
    1. Progress: Efficiently moving along the intended path.
    2. Safety: Maintaining a safe distance from other agents.
    3. Comfort: Executing smooth, comfortable maneuvers.
    4. Rule-Following: Adhering to lane boundaries and traffic signals.

    Args:
        state: The flat state tensor for the current timestep, shape (B, D_s).
        action: The action tensor, shape (B, D_a).
        next_state: The flat state tensor for the next timestep, shape (B, D_s).
        done: The done tensor, shape (B, 1).
        device: The torch device ('cuda' or 'cpu').

    Returns:
        A scalar reward tensor of shape (B, 1).
    """
    # batch_size = state.shape[0]
    
    # --- 0. Reconstruct the structured state for easier access ---
    # We only need to reconstruct the current state for most calculations
    # state_dict = reconstructor(state)
    ego_state = state['ego']
    agents_state = state['agents']
    agents_mask = state['agents_mask']
    tl_state = state['traffic_lights']

    # We need to reconstruct the next state to calculate jerk
    # next_state_dict = reconstructor(next_state)
    next_ego_state = next_state['ego']

    # --- 1. Progress Reward (Positive) ---
    # Reward for forward speed.
    speed = ego_state[:, 0] # Ego feature is just speed
    progress_reward = speed / 20.0 # Normalize by a reasonable max speed (20 m/s)

    # --- 2. Safety Penalty (Large Negative) ---
    # Penalty for being too close to other agents.
    # agents_state has shape (B, N, 7), where features are [x, y, vx, vy, cos(h), sin(h), valid]
    agent_positions = agents_state[:, :, :2] # Shape: (B, N, 2)
    distances = torch.linalg.norm(agent_positions, dim=-1) # Distance from ego (0,0)
    
    safety_margin = 2.0 # meters
    # Penalty increases exponentially as distance drops below the margin
    # We use a softplus-like function for a smooth penalty.
    # Inverse distance makes the penalty very high when close.
    # The mask ensures we only consider valid agents.
    proximity_penalty = torch.sum(
        torch.exp(-distances / safety_margin) * agents_mask, 
        dim=1
    )
    
    # --- 3. Comfort Penalty (Small Negative) ---
    # a) Penalty for high acceleration/deceleration (from the action itself)
    accel_penalty = (action[:, 0].abs() / 8.0).pow(2) # Penalize accel > 8 m/s^2

    # b) Penalty for high jerk (change in speed)
    # We can approximate this from the change in the 'speed' ego feature
    next_speed = next_ego_state[:, 0]
    jerk = (next_speed - speed) / 0.1 # Change in speed over one timestep
    jerk_penalty = (jerk / 10.0).pow(2) # Penalize jerk > 10 m/s^3

    # --- 4. Rule-Following / Lane Adherence Penalty (Moderate Negative) ---
    # a) Penalty for red lights
    # tl_state is [is_red_ahead, dist_to_stop_line]
    is_red = tl_state[:, 0]
    dist_to_red = tl_state[:, 1]
    # Heavy penalty if moving towards a close red light
    red_light_penalty = is_red * (speed > 1.0) * torch.exp(-dist_to_red / 10.0)

    # b) Lane Deviation Penalty - THIS IS A BIT TRICKY
    # The flat state vector does not contain the lane deviation info.
    # We would need to either:
    #   A) Add it to the state vector (modify FeatureExtractor)
    #   B) Re-calculate it here (very slow, requires map data)
    #   C) Omit it for now.
    # Let's go with C for simplicity and speed. The offline data already
    # strongly encourages lane following.

    # --- 5. Combine with Weights ---
    # These weights can be tuned. Let's start with a reasonable set.
    w_progress = 1.0
    w_safety = -2.0   # Safety is very important
    w_accel = -0.1
    w_jerk = -0.2     # Jerk is often more uncomfortable than pure accel
    w_red_light = -5.0 # Violating a red light is a critical failure

    final_reward = (w_progress * progress_reward + 
                    w_safety * proximity_penalty +
                    w_accel * accel_penalty + 
                    w_jerk * jerk_penalty +
                    w_red_light * red_light_penalty)
    
    return final_reward.unsqueeze(1) # Return as shape (B, 1)

def compute_reward_v3(
    state_dict: Dict[str, torch.Tensor], 
    action: torch.Tensor, 
    next_state_dict: Dict[str, torch.Tensor], 
    done: torch.Tensor,
    config: Dict
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]: # <-- CHANGED RETURN TYPE
    """
    (V3 - Goal-Conditioned, Structured) A professional, multi-objective reward function.
    
    This function computes a scalar reward as a weighted sum of components related to:
    1. Goal Progress: Efficiently making progress towards the planned route.
    2. Safety: Avoiding collisions with other agents.
    3. Comfort: Executing smooth, comfortable maneuvers.
    4. Path Adherence: Staying centered in the lane and following the intended path.
    5. Rule-Following: Obeying traffic signals.
    """
    
    # --- 0. Unpack relevant states ---
    ego_state = state_dict['ego']
    next_ego_state = next_state_dict['ego']
    agents_state = state_dict['agents']
    agents_mask = state_dict['agents_mask']
    tl_state = state_dict['traffic_lights']
    goal_state = state_dict['goal'] # <-- NEW
    goal_mask = state_dict['goal_mask'] # <-- NEW

    speed = ego_state[:, 0]
    batch_size = speed.shape[0]

    # --- 1. Goal Progress Reward (Major Improvement) ---
    # We want to reward the SDC for reducing its distance to the final goal point.
    
    # Find the furthest valid goal point for each sample in the batch
    # goal_state is (B, num_goals, 2), goal_mask is (B, num_goals)
    last_valid_goal_idx = goal_mask.long().sum(dim=1) - 1
    # Use a tensor of zeros for samples with no valid goal points
    final_goal_points = torch.zeros(batch_size, 2, device=speed.device)
    
    # Create a mask for batches that have at least one valid goal
    has_goal_mask = last_valid_goal_idx >= 0
    if has_goal_mask.any():
        # Gather the final goal points using the indices
        final_goal_points[has_goal_mask] = goal_state[has_goal_mask, last_valid_goal_idx[has_goal_mask]]
    
    # The progress is the reduction in distance to this final goal point.
    # We can approximate this by rewarding speed directed towards the goal.
    # Vector from ego (0,0) to the goal point
    goal_vector = final_goal_points 
    goal_direction = goal_vector / (torch.linalg.norm(goal_vector, dim=1, keepdim=True) + 1e-6)
    
    # SDC's velocity vector in the ego frame is approximately (speed, 0)
    ego_velocity_vector = torch.stack([speed, torch.zeros_like(speed)], dim=1)
    
    # Project ego velocity onto the goal direction vector (dot product)
    progress_reward = torch.einsum('bi,bi->b', ego_velocity_vector, goal_direction)
    # Normalize by max speed to keep it in a reasonable range
    progress_reward = progress_reward / 20.0

    # --- 2. Safety Penalty (More Robust) ---
    agent_positions = agents_state[:, :, :2]
    distances = torch.linalg.norm(agent_positions, dim=-1)
    
    safety_threshold = 3.0 # meters
    # A simple, quadratic penalty is more stable than exponential
    # Penalty = (1 - dist/thresh)^2 for dist < thresh
    proximity_penalty = torch.sum(
        (1.0 - distances / safety_threshold).relu().pow(2) * agents_mask, 
        dim=1
    )

    # --- 3. Comfort Penalty (Same logic) ---
    accel_penalty = (action[:, 0].abs() / 8.0).pow(2)
    next_speed = next_ego_state[:, 0]
    jerk = (next_speed - speed) / 0.1
    jerk_penalty = (jerk / 10.0).pow(2)

    # --- 4. Path Adherence Penalty (NEW and CRITICAL) ---
    # We can approximate lane deviation by looking at the *first* goal point,
    # which should be on the lane center ahead of us.
    first_goal_point = goal_state[:, 0, :] # Shape (B, 2)
    # The lateral deviation is simply the y-coordinate of the first goal point
    lane_deviation_penalty = (first_goal_point[:, 1].abs() / 1.5).pow(2) # Penalize deviation > 1.5m

    # --- 5. Rule-Following Penalty (Same logic) ---
    is_red = tl_state[:, 0]
    dist_to_red = tl_state[:, 1]
    red_light_penalty = is_red * (speed > 1.0) * torch.exp(-dist_to_red / 10.0)

    # --- 6. Combine with Tunable Weights ---
    # These should be moved to the config file for easy tuning
    reward_components = {
        'r_progress': progress_reward,
        'r_safety_prox': proximity_penalty,
        'r_comfort_accel': accel_penalty,
        'r_comfort_jerk': jerk_penalty,
        'r_adherence_lane': lane_deviation_penalty,
        'r_rule_red_light': red_light_penalty
    }

    # --- Combine with Weights ---
    w = config['reward']['weights_v3'] # Assume weights are in a sub-dict

    final_reward = (w['progress'] * reward_components['r_progress'] + 
                    w['safety'] * reward_components['r_safety_prox'] +
                    w['comfort_accel'] * reward_components['r_comfort_accel'] + 
                    w['comfort_jerk'] * reward_components['r_comfort_jerk'] +
                    w['adherence_lane'] * reward_components['r_adherence_lane'] +
                    w['rule_red_light'] * reward_components['r_rule_red_light'])
    
    # Return both the final reward and the dictionary of components
    return final_reward.unsqueeze(1), reward_components