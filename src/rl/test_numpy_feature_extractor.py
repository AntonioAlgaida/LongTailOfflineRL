# tests/test_feature_extractor.py

# To run:
# conda activate wwm
# python -m src.rl.test_feature_extractor

import os
import sys
import numpy as np
from glob import glob
import random
import traceback

# Adjust path to go up one more level for the tests/ directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.rl.feature_extractor import FeatureExtractor

def main():
    print("--- Testing Simplified FeatureExtractor (V2) ---")
    
    # 1. Load Config and instantiate FeatureExtractor
    config = load_config()
    feature_extractor = FeatureExtractor(config)
    
    # 2. Load a random scenario file
    validation_npz_dir = os.path.join(config['data']['processed_npz_dir'], 'validation')
    all_files = glob(os.path.join(validation_npz_dir, '*.npz'))
    if not all_files:
        print(f"❌ Error: No .npz files found in {validation_npz_dir}. Please run the parser first.")
        return
        
    random_npz_file = random.choice(all_files)
    print(f"\nLoading random scenario: {os.path.basename(random_npz_file)}")
    scenario_data = np.load(random_npz_file, allow_pickle=True)
    
    # 3. Call extract_features for a valid timestep
    # Find a valid timestep for the SDC to ensure the extractor runs
    sdc_idx = scenario_data['sdc_track_index']
    valid_timesteps = np.where(scenario_data['valid_mask'][sdc_idx, :])[0]
    if len(valid_timesteps) == 0:
        print(f"❌ Scenario {os.path.basename(random_npz_file)} has no valid SDC timesteps. Skipping.")
        return
        
    timestep_to_test = random.choice(valid_timesteps)
    print(f"Extracting features for a valid timestep: {timestep_to_test}\n")
    
    try:
        # The new extractor returns a single flat vector
        state_vector = feature_extractor.extract_features(scenario_data, timestep_to_test)
        
        # 4. Validation Checks
        print("--- Validation Results ---")
        
        # --- Check 1: Validate Overall Shape ---
        cfg_features = config['features']
        
        # Calculate the expected dimension from the config
        expected_ego_dim = 1
        expected_agent_dim = cfg_features['num_agents'] * 7
        expected_map_dim = cfg_features['num_map_polylines'] * (cfg_features['map_points_per_polyline'] * 2 + 1)
        expected_tl_dim = 2
        
        expected_total_dim = expected_ego_dim + expected_agent_dim + expected_map_dim + expected_tl_dim
        
        print(f"Output vector shape: {state_vector.shape}")
        print(f"Expected total dimension: {expected_total_dim}")
        assert state_vector.shape == (expected_total_dim,), "❌ Shape mismatch!"
        print("✅ Overall feature vector shape is correct.")

        # --- Check 2: Validate Data Type ---
        print(f"\nData type: {state_vector.dtype}")
        assert state_vector.dtype == np.float32, "❌ Data type is not float32!"
        print("✅ Data type is correct.")

        # --- Check 3: Validate Numerical Stability (CRITICAL) ---
        has_nan = np.isnan(state_vector).any()
        has_inf = np.isinf(state_vector).any()
        
        print(f"\nContains NaN values: {has_nan}")
        print(f"Contains Inf values: {has_inf}")
        assert not has_nan and not has_inf, "❌ Invalid number (NaN or Inf) found in state vector!"
        print("✅ State vector is numerically stable (no NaNs or Infs).")

        # --- Check 4: Print Statistical Summary ---
        mean_val = np.mean(state_vector)
        std_val = np.std(state_vector)
        min_val = np.min(state_vector)
        max_val = np.max(state_vector)
        
        print("\n--- Statistical Summary ---")
        print(f"  Min value: {min_val:.4f}")
        print(f"  Max value: {max_val:.4f}")
        print(f"  Mean value: {mean_val:.4f}")
        print(f"  Std Dev: {std_val:.4f}")
        
        # A simple sanity check on the scale of the values
        assert max_val < 1000 and min_val > -1000, "❌ Values seem unusually large, check for errors."
        print("✅ Feature values are within a reasonable range.")
        
        print("\n\n✅✅✅ All Tests Passed: Simplified FeatureExtractor is validated! ✅✅✅")
        
    except Exception as e:
        traceback.print_exc()
        print(f"\n❌ Test Failed: An exception occurred: {e}")

if __name__ == '__main__':
    main()