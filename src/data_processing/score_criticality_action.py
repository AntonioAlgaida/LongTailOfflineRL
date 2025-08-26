# src/data_processing/score_criticality_action.py

# Calculates timestep-level criticality scores based on the rarity of the expert's action.
# Run AFTER featurizer.py and compute_action_weights.py.
# To run:
# conda activate longtail-rl
# python -m src.data_processing.score_criticality_action

import os
import sys
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config

# --- Global Vars for Workers ---
CONFIG = None
ACTION_WEIGHT_DATA = None

def init_worker(config_path: str):
    """Initializer for each worker process to load config and action weights."""
    global CONFIG, ACTION_WEIGHT_DATA
    CONFIG = load_config(config_path)
    
    # Load the action weights data once per worker for efficiency
    weights_path = os.path.join(os.path.dirname(CONFIG['data']['feature_stats_path']), 'action_weights_v2.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Action weights not found at '{weights_path}'. "
                                "Please run compute_action_weights.py first.")
    ACTION_WEIGHT_DATA = torch.load(weights_path)

def process_shard(pt_path: str) -> bool:
    """
    Processes one pre-featurized .pt file, scores each action's rarity,
    and saves the scores to a new .npz file.
    """
    try:
        data = torch.load(pt_path, weights_only=False)

        scenario_id = os.path.splitext(os.path.basename(pt_path))[0]
        
        actions = np.stack(
                [sample['action'] for sample in data], 
                axis=0
            ) # Final shape: (num_samples_in_scenario, 2)
                    
        if actions.shape[0] == 0:
            return False # Skip empty scenarios

        # Get bin definitions and weights from the globally loaded data
        accel_bins = ACTION_WEIGHT_DATA['accel_bins']
        steer_bins = ACTION_WEIGHT_DATA['steer_bins']
        weights = ACTION_WEIGHT_DATA['weights']
        
        # --- Vectorized Score Calculation ---
        # Determine the bin index for each action in the scenario
        accel_indices = torch.bucketize(torch.tensor(actions[:, 0]), accel_bins) - 1
        steer_indices = torch.bucketize(torch.tensor(actions[:, 1]), steer_bins) - 1
        
        # Clamp indices to be within the valid range of the weights tensor
        accel_indices = torch.clamp(accel_indices, 0, weights.shape[0] - 1)
        steer_indices = torch.clamp(steer_indices, 0, weights.shape[1] - 1)
        
        # Look up the pre-computed rarity score (weight) for each action
        action_rarity_scores = weights[accel_indices, steer_indices].numpy()
        
        # --- Normalize and Save ---
        # The weights from compute_action_weights are already normalized around a mean of 1.
        # We can clip them to a max value to prevent extreme scores from dominating.
        # A value of 5 means the action is 5x rarer than average.
        final_scores = np.clip(action_rarity_scores, 0, 3.0) / 3.0
        final_scores += 0.01 # Add base value

        # Save scores to the correct subdirectory
        output_subdir = os.path.basename(os.path.dirname(pt_path))
        output_dir = os.path.join(CONFIG['data']['criticality_scores_dir_v2'], 'timestep_level', 'action_rarity', output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{scenario_id}.npz")
        np.savez_compressed(output_path, action_rarity=final_scores.astype(np.float32))
        
        return True
        
    except Exception:
        return False

def main():
    """Main orchestrator for the action rarity scoring pipeline."""
    config_path = os.path.join(PROJECT_ROOT, 'configs/main_config.yaml')
    global CONFIG
    CONFIG = load_config(config_path)

    output_dir = os.path.join(CONFIG['data']['criticality_scores_dir_v2'], 'timestep_level', 'action_rarity')
    scores_base_dir = CONFIG['data']['featurized_dir_v2']
    all_pt_paths = glob(os.path.join(scores_base_dir, '*', '*.pt'))
    
    if not all_pt_paths:
        print(f"Error: No featurized .pt files found in {scores_base_dir}.")
        return

    # --- Definitive Safe Deletion / Incremental Processing Logic ---
    paths_to_process = all_pt_paths

    if os.path.exists(output_dir):
        print(f"\nOutput directory '{output_dir}' already exists.")
        response = input("Choose an action: [d]elete and start fresh, [c]ontinue (skip existing), or [a]bort? [d/c/a]: ")
        
        if response.lower() == 'd':
            print(f"Deleting existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
            print("Directory deleted.")
            # The workers will recreate the necessary subdirectories.
        
        elif response.lower() == 'c':
            print("Continuing. Will skip files that have already been processed.")
            # Get a set of scenario IDs that already exist in the output directory
            existing_ids = set()
            # We need to scan both potential subdirectories ('training' and 'validation')
            for subdir in ['training', 'validation']:
                path = os.path.join(output_dir, subdir)
                if os.path.exists(path):
                    # Get the filename without any extension (handles .npz, .pt, etc.)
                    existing_ids.update(
                        [f.split('.')[0] for f in os.listdir(path)]
                    )

            # Filter the list of input .pt paths to only include those not yet processed
            paths_to_process = [
                p for p in all_pt_paths
                if os.path.basename(p).split('.')[0] not in existing_ids
            ]
            print(f"Found {len(existing_ids)} already scored scenarios. Skipping them.")
            
        else:
            print("Aborting.")
            return
    
    if not paths_to_process:
        print("\nAll scenarios have already been scored. Nothing to do.")
        return

    print(f"\nFound {len(paths_to_process)} new scenarios to score (out of {len(all_pt_paths)} total).")
    
    num_workers = CONFIG['data'].get('num_workers', cpu_count())
    print(f"Using {num_workers} worker processes.")

    with Pool(processes=num_workers, initializer=init_worker, initargs=(config_path,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_shard, all_pt_paths), total=len(all_pt_paths)))

    success_count = sum(results)
    print("\n-------------------------------------------")
    print("Action-Rarity Scoring complete!")
    print(f"Successfully processed and saved scores for {success_count} scenarios.")
    print(f"Failed or empty scenarios: {len(results) - success_count}")
    print(f"Output saved to: {output_dir}")
    print("-------------------------------------------")

if __name__ == '__main__':
    main()