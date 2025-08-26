# src/data_processing/compute_action_weights.py

# Calculates action frequency weights for a weighted MSE loss.
# Should be run after featurizer.py and before train_scout_ensemble.py
# To run:
# conda activate wwm
# python -m src.data_processing.compute_action_weights

import os
import sys
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import traceback

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config

# --- Worker Function for Parallel Processing ---
def process_chunk(chunk_data: tuple) -> np.ndarray:
    """
    Worker function. Processes a list (chunk) of .pt files and returns
    a local histogram of action frequencies.
    """
    file_chunk, accel_bins, steer_bins = chunk_data
    
    local_histogram = np.zeros((len(accel_bins) - 1, len(steer_bins) - 1), dtype=np.int64)
    
    for pt_path in file_chunk:
        try:
            # Load the pre-featurized actions tensor
            scenario_samples = torch.load(pt_path, weights_only=False)
            
            if not scenario_samples:
                print(f"Warning: No samples found in {pt_path}. Skipping.")
                continue

            # --- EFFICIENT BATCH EXTRACTION ---
            # 1. Extract all 'action' arrays from the list of dicts
            #    and stack them into a single NumPy array.
            #    This is a fast list comprehension.
            all_actions_in_scenario = np.stack(
                [sample['action'] for sample in scenario_samples], 
                axis=0
            ) # Final shape: (num_samples_in_scenario, 2)
            # --- END OF EFFICIENT EXTRACTION ---

            # Now, your existing vectorized logic will work perfectly.
            accel_vals = all_actions_in_scenario[:, 0]
            steer_vals = all_actions_in_scenario[:, 1]
            
            accel_binned = np.digitize(accel_vals, accel_bins) - 1
            steer_binned = np.digitize(steer_vals, steer_bins) - 1
            
            valid_mask = (accel_binned >= 0) & (accel_binned < local_histogram.shape[0]) & \
                         (steer_binned >= 0) & (steer_binned < local_histogram.shape[1])
            
            np.add.at(local_histogram, (accel_binned[valid_mask], steer_binned[valid_mask]), 1)

        except Exception as e:
            print(f"Error processing {pt_path}: {e}")
            traceback.print_exc()
            continue
            
    return local_histogram

# --- Main Orchestrator ---
def main():
    config = load_config()
    num_workers = config['data'].get('num_workers', cpu_count())
    print(f"--- Computing Action Frequency Weights using {num_workers} cores ---")

    # --- 1. Define the Non-Uniform Discretization Grid ---
    # These bins are defined based on the action space limits in the config
    action_config = config['action_space']
    min_accel = action_config['min_acceleration'] # -10.0
    max_accel = action_config['max_acceleration'] # 8.0
    max_steer = action_config['max_yaw_rate']     # 1.0 (since yaw rate is our steering)

    ACCEL_BINS = np.concatenate([
        np.linspace(min_accel, -2.0, num=20),     # Braking
        np.linspace(-2.0, 1.5, num=50),          # Low-intensity control
        np.linspace(1.5, max_accel, num=20)      # Acceleration
    ])

    STEER_BINS = np.concatenate([
        np.linspace(-max_steer, -0.2, num=25),   # Left turns
        np.linspace(-0.2, 0.2, num=40),          # Driving straight (high resolution)
        np.linspace(0.2, max_steer, num=25)      # Right turns
    ])
    
    print(f"Using a grid of {len(ACCEL_BINS)-1} accel bins and {len(STEER_BINS)-1} steering bins.")
    
    # --- 2. Get File List and Split into Chunks ---
    train_data_dir = os.path.join(config['data']['featurized_dir_v2'], 'training')
    data_files = glob(os.path.join(train_data_dir, '*.pt'))
    if not data_files:
        raise FileNotFoundError(f"No featurized .pt files found in '{train_data_dir}'. Please run featurizer.py first.")
        
    print(f"Found {len(data_files)} featurized scenario files to analyze.")
    
    file_chunks = np.array_split(data_files, num_workers * 4) # More chunks for a smoother progress bar
    tasks = [(chunk, ACCEL_BINS, STEER_BINS) for chunk in file_chunks]

    # --- 3. Run Multiprocessing and Combine Results ---
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_chunk, tasks), total=len(tasks), desc="Processing chunks"))
    
    final_histogram = np.sum(results, axis=0)
            
    # --- 4. Calculate Smoothed Inverse Frequency Weights ---
    print("Calculating inverse frequency weights...")
    epsilon = 1.0 # Add-one smoothing
    action_weights = 1.0 / (final_histogram.astype(np.float32) + epsilon)
    # Normalize weights so that their mean is 1.0, preserving the loss scale
    action_weights /= np.mean(action_weights)

    # --- 5. Save the Weight Matrix and Bins ---
    output_path = os.path.join(os.path.dirname(config['data']['feature_stats_path']), 'action_weights_v2.pt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    weight_data = {
        'weights': torch.from_numpy(action_weights),
        'accel_bins': torch.from_numpy(ACCEL_BINS),
        'steer_bins': torch.from_numpy(STEER_BINS) # Steer bins are yaw_rate bins
    }
    torch.save(weight_data, output_path)
    
    print(f"\n--- Weight computation complete ---")
    print(f"Action weights saved to: {output_path}")

if __name__ == "__main__":
    main()