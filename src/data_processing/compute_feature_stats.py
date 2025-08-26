# src/data_processing/compute_feature_stats.py

# This script runs AFTER featurizer.py.
# It reads the final .pt files and calculates normalization statistics.
# To run:
# conda activate wwm
# python -m src.data_processing.compute_feature_stats

import os
import sys
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config

# --- Welford's Algorithm for Stable Online Statistics ---
class RunningStats:
    """
    (Robust Version) Calculates mean and variance in a single pass using
    Welford's online algorithm. It is numerically stable and handles NaNs.
    It can be updated with single values or entire batches of data.
    """
    def __init__(self, shape: tuple):
        self.shape = shape
        self.count = np.zeros(shape, dtype=np.int64)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64) # Sum of squared differences

    def update(self, new_data: np.ndarray):
        """Updates stats with a new batch of data, ignoring NaNs."""
        if new_data.ndim == len(self.shape): # Single sample
            new_data = new_data.reshape(1, *self.shape)
        
        for x in new_data:
            valid_mask = ~np.isnan(x)
            self.count[valid_mask] += 1
            
            delta = x[valid_mask] - self.mean[valid_mask]
            self.mean[valid_mask] += delta / self.count[valid_mask]
            delta2 = x[valid_mask] - self.mean[valid_mask]
            self.m2[valid_mask] += delta * delta2
            
            # Print a warning if any nan has been found
            if np.any(np.isnan(x)):
                print(f"Warning: NaN found in input data for feature '{self.shape}'")

    def get_stats(self) -> tuple:
        """Returns the final mean and standard deviation."""
        safe_count = np.maximum(self.count, 2)
        variance = self.m2 / (safe_count - 1)
        std_dev = np.sqrt(variance)
        
        mean_final = np.nan_to_num(self.mean, nan=0.0)
        std_final = np.nan_to_num(std_dev, nan=1.0) # Default std to 1.0 for safety
        return mean_final, std_final

# --- MapReduce Functions for Parallelization ---
def combine_stats(stat_list: List[Dict[str, Tuple]]) -> Dict[str, RunningStats]:
    """(Reduce Step) Combines stats from multiple workers."""
    stat_list = [s for s in stat_list if s]
    if not stat_list: return {}
    
    first_worker_stats = stat_list[0]
    combined_trackers = {key: RunningStats(shape=s[1].shape) for key, s in first_worker_stats.items()}
    for key in combined_trackers:
        count, mean, m2 = first_worker_stats[key]
        combined_trackers[key].count, combined_trackers[key].mean, combined_trackers[key].m2 = count, mean, m2

    for worker_stats in stat_list[1:]:
        for key, tracker in combined_trackers.items():
            count_b, mean_b, m2_b = worker_stats[key]
            if np.sum(count_b) == 0: continue
            
            count_a, mean_a, m2_a = tracker.count, tracker.mean, tracker.m2
            new_count = count_a + count_b
            safe_mask = new_count > 0
            delta = mean_b - mean_a
            
            new_mean, new_m2 = mean_a.copy(), m2_a.copy()
            new_mean[safe_mask] = mean_a[safe_mask] + delta[safe_mask] * (count_b[safe_mask] / new_count[safe_mask])
            new_m2[safe_mask] = m2_a[safe_mask] + m2_b[safe_mask] + \
                                delta[safe_mask]**2 * (count_a[safe_mask] * count_b[safe_mask] / new_count[safe_mask])
            
            tracker.count, tracker.mean, tracker.m2 = new_count, new_mean, new_m2
            
    return combined_trackers

def process_files_worker(file_list: List[str]) -> Dict[str, Tuple]:
    """
    (Map Step) Worker function that processes structured, pre-featurized .pt files.
    """
    stats_trackers = {}
    initialized = False

    for pt_path in file_list:
        try:
            # Each .pt file contains a list of sample dictionaries
            scenario_samples = torch.load(pt_path, weights_only=False)
            if not scenario_samples:
                continue

            # --- Initialize trackers on the first valid sample ---
            if not initialized:
                first_sample = scenario_samples[0]
                # Create a tracker for the 'action' tensor
                stats_trackers['action'] = RunningStats(shape=first_sample['action'].shape)
                # Create a tracker for each tensor within the 'state' dictionary
                for key, tensor in first_sample['state'].items():
                    # We don't need stats for boolean masks
                    if not key.endswith('_mask'):
                        stats_trackers[key] = RunningStats(shape=tensor.shape)
                initialized = True

            # --- Update stats for all samples in the scenario ---
            for sample in scenario_samples:
                stats_trackers['action'].update(sample['action'])
                for key, tensor in sample['state'].items():
                    if key in stats_trackers:
                        stats_trackers[key].update(tensor)
        except Exception:
            continue
            
    if not initialized:
        return {}
        
    # Return the raw statistics needed for the combine step
    return {key: (tracker.count, tracker.mean, tracker.m2) for key, tracker in stats_trackers.items()}

# --- Main Orchestrator ---
def main():
    print("--- Starting Parallel Feature Statistics Calculation (for Structured V2 Data) ---")
    config = load_config()
    
    # Read from the new featurized_dir_v2
    train_data_dir = os.path.join(config['data']['featurized_dir_v2'], 'training')
    all_pt_files = glob(os.path.join(train_data_dir, '*.pt'))

    if not all_pt_files:
        print(f"❌ Error: No featurized V2 .pt files found in '{train_data_dir}'. Please run the new featurizer.py first.")
        return
        
    print(f"Found {len(all_pt_files)} featurized scenario files in the training set.")
    
    num_workers = config['data'].get('num_workers', cpu_count())
    print(f"Using {num_workers} worker processes.")

    # Split the file list into chunks for each worker
    file_chunks = np.array_split(all_pt_files, num_workers * 4) # More chunks for a smoother progress bar
    
    # Map Step
    print("\n[Map Step] Processing file chunks in parallel...")
    with Pool(processes=num_workers) as pool:
        worker_results = list(tqdm(pool.imap(process_files_worker, file_chunks), total=len(file_chunks)))

    # Reduce Step
    print("\n[Reduce Step] Combining results from all workers...")
    final_trackers = combine_stats(worker_results)
    
    if not final_trackers:
        print("❌ Error: No data was processed successfully.")
        return

    # Finalize and Save
    print("\nCalculation complete. Finalizing and saving statistics...")
    final_stats = {}
    # The final trackers dict now has keys like 'ego', 'agents', 'goal', 'action'
    for key, tracker in final_trackers.items():
        mean, std = tracker.get_stats()
        final_stats[f'{key}_mean'] = mean.astype(np.float32)
        final_stats[f'{key}_std'] = std.astype(np.float32)
        print(f"  - Stats for '{key}': shape={mean.shape}, mean_range=({mean.min():.2f}, {mean.max():.2f}), std_range=({std.min():.2f}, {std.max():.2f})")

    # Save to the new versioned path
    output_path = config['data']['feature_stats_path_v2']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **final_stats)

    print("\n-------------------------------------------")
    print("✅ V2 Feature statistics saved successfully!")
    print(f"Output file: {output_path}")
    print("-------------------------------------------")
    
if __name__ == '__main__':
    main()