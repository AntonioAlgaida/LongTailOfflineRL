# tests/test_reconstructor.py

# To run:
# conda activate longtail-rl
# python -m tests.test_reconstructor

import os
import sys
import torch
from glob import glob
import random

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.rl.networks import StateReconstructor

def test_reconstructor_on_real_data():
    """
    Tests the StateReconstructor using a real data file produced by the
    featurizer.py script. This provides an end-to-end validation of the
    data shape pipeline.
    """
    print("--- Testing StateReconstructor on Real Data ---")
    
    # --- 1. Setup ---
    config = load_config()
    reconstructor = StateReconstructor(config)
    
    # --- 2. Load a random pre-featurized data file ---
    featurized_dir = os.path.join(config['data']['featurized_dir'], 'validation')
    all_pt_files = glob(os.path.join(featurized_dir, '*.pt'))
    if not all_pt_files:
        print(f"❌ Error: No pre-featurized .pt files found in '{featurized_dir}'.")
        print("Please run the featurizer.py script first.")
        return
        
    random_pt_file = random.choice(all_pt_files)
    print(f"\nLoading real data from: {os.path.basename(random_pt_file)}")
    
    try:
        data = torch.load(random_pt_file)
        flat_states_batch = data['states']
    except Exception as e:
        print(f"❌ Error loading data file: {e}")
        return

    # The "batch size" for this test is the number of valid timesteps in the scenario.
    if flat_states_batch.dim() != 2 or flat_states_batch.shape[0] == 0:
        print(f"Skipping test: Scenario file {os.path.basename(random_pt_file)} contains no valid samples or has incorrect dimensions.")
        return
    
    batch_size = flat_states_batch.shape[0] # e.g., 90
        
    print(f"Using a batch of size {batch_size} (all timesteps) from the file.")
    print(f"Shape of flat state tensor: {flat_states_batch.shape}")

    # --- 3. Run the reconstruction ---
    state_dict = reconstructor(flat_states_batch)
    print("Reconstruction successful. Output dictionary keys:", state_dict.keys())
    
    # --- 4. Validate the output ---
    cfg_features = config['features']
    n_agents = cfg_features['num_agents']
    n_map = cfg_features['num_map_polylines']
    ego_dim, agent_feature_dim, map_feature_dim, tl_dim = 1, 7, (cfg_features['map_points_per_polyline'] * 2) + 1, 2

    # Check 4a: Keys
    expected_keys = {'ego', 'agents', 'agents_mask', 'map', 'map_mask', 'traffic_lights'}
    assert set(state_dict.keys()) == expected_keys, "Incorrect keys in output dict."
    print("✅ Output dictionary contains the correct keys.")
    
    # Check 4b: Shapes
    assert state_dict['ego'].shape == (batch_size, ego_dim), "Incorrect shape for 'ego'"
    assert state_dict['agents'].shape == (batch_size, n_agents, agent_feature_dim), "Incorrect shape for 'agents'"
    assert state_dict['agents_mask'].shape == (batch_size, n_agents), "Incorrect shape for 'agents_mask'"
    assert state_dict['map'].shape == (batch_size, n_map, map_feature_dim), "Incorrect shape for 'map'"
    assert state_dict['map_mask'].shape == (batch_size, n_map), "Incorrect shape for 'map_mask'"
    assert state_dict['traffic_lights'].shape == (batch_size, tl_dim), "Incorrect shape for 'traffic_lights'"
    print("✅ All tensors in the output dictionary have the correct shapes.")
    
    # Check 4c: Numerical Stability
    for key, tensor in state_dict.items():
        assert not torch.isnan(tensor).any(), f"NaN found in tensor '{key}'"
        assert not torch.isinf(tensor).any(), f"Inf found in tensor '{key}'"
    print("✅ All reconstructed tensors are numerically stable (no NaNs or Infs).")
    
    print("\n--- All checks passed! ---")
    print("\n✅✅✅ StateReconstructor is validated on real data! ✅✅✅")

if __name__ == '__main__':
    test_reconstructor_on_real_data()