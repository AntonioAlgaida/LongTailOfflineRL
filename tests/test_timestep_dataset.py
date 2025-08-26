# tests/test_timestep_dataset.py

# To run:
# conda activate longtail-rl
# python -m tests.test_timestep_dataset

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from glob import glob
import random

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.rl.dataset import OfflineRLTimestepDataset
from src.rl.networks import StateReconstructor

def test_timestep_dataset_with_real_data():
    """
    An integration test for the OfflineRLTimestepDataset.
    
    It uses a small subset of the real featurized data and real score files
    to verify:
    1. Instantiation and correct indexing (handling contiguous transitions).
    2. The critical `load_and_align_weights` method.
    3. Compatibility with `WeightedRandomSampler` and `DataLoader`.
    """
    print("--- Running Integration Test for OfflineRLTimestepDataset ---")
    
    # --- 1. Setup ---
    config = load_config()
    reconstructor = StateReconstructor(config)
    state_dim = reconstructor.end_tl
    
    # --- Use a small subset of REAL validation data for the test ---
    featurized_dir = os.path.join(config['data']['featurized_dir'], 'validation')
    all_pt_files = sorted(glob(os.path.join(featurized_dir, '*.pt')))
    
    if len(all_pt_files) < 5:
        print("❌ Test failed: Not enough featurized data files found in validation set for a meaningful test.")
        return
        
    # Use just the first 5 files for a quick but realistic test
    test_file_paths = all_pt_files[:5]
    print(f"Using a subset of {len(test_file_paths)} real data files for testing.")

    # --- 2. Test Instantiation and Indexing ---
    try:
        dataset = OfflineRLTimestepDataset(test_file_paths)
        print("\n✅ Dataset instantiated successfully.")
        assert len(dataset) > 0, "Dataset should have found some valid transitions."
        print(f"✅ Dataset indexing is working. Found {len(dataset)} contiguous transitions.")
        
    except Exception as e:
        print(f"❌ Test failed during instantiation or indexing: {e}")
        return

    # --- 3. Test the `load_and_align_weights` method ---
    # We will test it with the heuristic scores
    print("\n--- Testing `load_and_align_weights` method ---")
    score_dir = os.path.join(config['data']['criticality_scores_dir'], 'timestep_level', 'heuristic')
    
    # Check if score files exist
    score_files_exist = glob(os.path.join(score_dir, '*', '*.npz'))
    if not score_files_exist:
        print(f"❌ Test failed: No heuristic score files found in '{score_dir}'.")
        print("Please run `score_criticality_heuristic.py` first.")
        return

    try:
        weights = dataset.load_and_align_weights(score_dir, config)
        
        # Check shape
        assert weights.shape == (len(dataset),), \
            f"Weights tensor shape is incorrect! Expected ({len(dataset)},), got {weights.shape}"
        print("✅ `load_and_align_weights` returned a tensor of the correct shape.")
        
        # Check type
        assert weights.dtype == torch.float32, "Weights tensor has incorrect dtype."
        print("✅ Weights tensor has the correct dtype.")

        # Check for non-uniformity (a simple sanity check)
        # If all weights are 1.0, it means no scores were found/aligned correctly.
        assert not torch.all(weights == 1.0), "Weights are all 1.0; alignment might have failed."
        print("✅ Weights appear to be correctly aligned (values are not uniform).")

    except Exception as e:
        print(f"❌ Test failed during weight loading/alignment: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Test DataLoader with WeightedRandomSampler ---
    print("\n--- Testing DataLoader with WeightedRandomSampler ---")
    try:
        # Create the sampler with the loaded weights
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        
        data_loader = DataLoader(
            dataset,
            batch_size=4,
            sampler=sampler,
            num_workers=0 # Use 0 workers for this test to avoid multiprocessing complexities
        )
        
        # Fetch a few batches to ensure it works
        num_batches_to_test = 5
        for i, batch in enumerate(data_loader):
            if i >= num_batches_to_test:
                break
            assert batch['states'].shape == (4, state_dim), f"Batch {i} has an incorrect shape."
            
        print(f"✅ Successfully retrieved {num_batches_to_test} batches using WeightedRandomSampler.")

    except Exception as e:
        print(f"❌ Test failed with DataLoader and WeightedRandomSampler: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- All checks passed! ---")
    print("\n✅✅✅ OfflineRLTimestepDataset is fully validated with real data and weights! ✅✅✅")

if __name__ == '__main__':
    test_timestep_dataset_with_real_data()