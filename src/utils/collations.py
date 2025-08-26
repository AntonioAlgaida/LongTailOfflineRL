# src/utils/collations.py

import torch
import numpy as np

def structured_collate_fn(batch: list) -> dict:
    """
    Custom collate function for the OfflineRLDataset.
    
    It takes a list of sample dictionaries (each containing a nested state dict)
    and collates them into a single batch dictionary of tensors.

    Args:
        batch: A list of sample dictionaries from the dataset.
               e.g., [{'states': {...}, 'actions': ...}, {'states': {...}, ...}]

    Returns:
        A single dictionary where each value is a batched tensor.
    """
    # --- Collate simple tensors first (actions, rewards, dones) ---
    actions = torch.from_numpy(np.array([d['actions'] for d in batch]))
    rewards = torch.cat([d['rewards'] for d in batch], dim=0)
    dones = torch.cat([d['dones'] for d in batch], dim=0)
    
    # --- Collate the nested state dictionaries ---
    states_list = [d['states'] for d in batch]
    next_states_list = [d['next_states'] for d in batch]
    
    collated_states = {}
    collated_next_states = {}
    
    # Get the keys from the first state dictionary (e.g., 'ego', 'agents', 'map', ...)
    state_keys = states_list[0].keys()
    
    for key in state_keys:
        # Stack the tensors for the current key from all samples in the batch
        collated_states[key] = torch.from_numpy(np.array([s[key] for s in states_list]))
        collated_next_states[key] = torch.from_numpy(np.array([s[key] for s in next_states_list]))
        
    return {
        'states': collated_states,
        'actions': actions,
        'rewards': rewards,
        'next_states': collated_next_states,
        'dones': dones
    }