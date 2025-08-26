# src/data_processing/score_criticality_ensemble.py

# Calculates timestep-level criticality scores based on ensemble disagreement.
# Run AFTER train_scout_ensemble.py.
# To run:
# conda activate longtail-rl
# python -m src.data_processing.score_criticality_ensemble

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
# We need the model definition to load the saved state_dicts
from src.data_processing.train_scout_ensemble import ScoutBCModel, structured_collate_fn

# --- Global Vars for Workers ---
CONFIG = None
SCOUT_MODELS = None
FLAT_MEAN = None
FLAT_STD = None
DEVICE = None

def init_worker(config_path: str):
    """
    Initializer for each worker process. Loads config, structured normalization
    stats, and all trained scout models into the worker's memory.
    """
    global CONFIG, SCOUT_MODELS, MEAN_DICT, STD_DICT, DEVICE
    
    CONFIG = load_config(config_path)
    # CPU is safer for this data processing task
    DEVICE = torch.device("cuda")

    # --- Load STRUCTURED Normalization Stats ---
    stats = np.load(CONFIG['data']['feature_stats_path_v2'])
    MEAN_DICT = {k.replace('_mean', ''): torch.from_numpy(v).to(DEVICE).float() for k, v in stats.items() if '_mean' in k}
    STD_DICT = {k.replace('_std', ''): torch.from_numpy(v).to(DEVICE).float() for k, v in stats.items() if '_std' in k}

    # --- Load all trained scout models ---
    SCOUT_MODELS = []
    model_dir = os.path.join('models', 'scout_ensemble')
    model_paths = sorted(glob(os.path.join(model_dir, '*.pth')))
    if not model_paths:
        raise FileNotFoundError(f"Scout models not found in '{model_dir}'.")
    
    for path in model_paths:
        # The ScoutBCModel now takes the config to initialize itself correctly
        model = ScoutBCModel(CONFIG)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        SCOUT_MODELS.append(model)
    
    print(f"Worker {os.getpid()} initialized with {len(SCOUT_MODELS)} models.")

def process_shard(pt_path: str) -> bool:
    """
    Processes one pre-featurized .pt file, calculates the ensemble disagreement
    score for each timestep, and saves the scores to a new .npy file.
    """
    try:
        scenario_samples = torch.loadc
        scenario_id = os.path.splitext(os.path.basename(pt_path))[0]
        
        if not scenario_samples:
            return False
            
        # --- Batch process the entire scenario for efficiency ---
        # 1. Collate all samples from the scenario into a single batch
        state_dicts, _ = structured_collate_fn(
            [(sample['state'], sample['action']) for sample in scenario_samples]
        )
        
        # 2. Move to device and normalize
        state_dicts = {k: v.to(DEVICE).float() for k, v in state_dicts.items()}
        for key in state_dicts:
            if not key.endswith('_mask'):
                state_dicts[key] = (state_dicts[key] - MEAN_DICT[key]) / (STD_DICT[key] + 1e-6)

        # 3. Get predictions from all models for all timesteps
        with torch.no_grad():
            all_predictions = [model(state_dicts) for model in SCOUT_MODELS]
        
        # Stack predictions: (num_models, num_timesteps, 2)
        predictions_tensor = torch.stack(all_predictions, dim=0)
        
        # Calculate variance across the 'models' dimension
        variance_per_timestep = torch.var(predictions_tensor, dim=0)
        
        # Disagreement score is the sum of variances (trace of covariance)
        disagreement_scores = torch.sum(variance_per_timestep, dim=1)
        
        # --- Normalize and Save ---
        scores_np = disagreement_scores.cpu().numpy()
        p99 = np.percentile(scores_np, 99)
        if p99 > 1e-6:
            final_scores = np.clip(scores_np / p99, 0, 1)
        else:
            final_scores = np.zeros_like(scores_np)

        # Save scores to the correct subdirectory
        output_subdir = os.path.basename(os.path.dirname(pt_path))
        output_dir = os.path.join(CONFIG['data']['criticality_scores_dir_v2'], 'timestep_level', 'ensemble', output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as .npz to be consistent with the heuristic scorer
        output_path = os.path.join(output_dir, f"{scenario_id}.npz")
        np.savez_compressed(output_path, ensemble=final_scores.astype(np.float32))
        
        return True
        
    except Exception:
        return False

def main():
    config_path = os.path.join(PROJECT_ROOT, 'configs/main_config.yaml')
    global CONFIG
    CONFIG = load_config(config_path)


    output_dir = os.path.join(CONFIG['data']['criticality_scores_dir_v2'], 'timestep_level', 'ensemble')
    scores_base_dir = CONFIG['data']['featurized_dir_v2']
    all_pt_paths = glob(os.path.join(scores_base_dir, '*', '*.pt'))
    
    if not all_pt_paths:
        print(f"Error: No featurized .pt files found in {scores_base_dir}.")
        print("Please run featurizer.py first.")
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

    # --- Now, the multiprocessing pool uses `paths_to_process` ---
    with Pool(processes=num_workers, initializer=init_worker, initargs=(config_path,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_shard, paths_to_process), total=len(paths_to_process)))

    success_count = sum(results)
    print("\n-------------------------------------------")
    print("Ensemble-based (Timestep) Scoring complete!")
    print(f"Successfully processed and saved scores for {success_count} scenarios.")
    print(f"Failed or empty scenarios: {len(results) - success_count}")
    print(f"Output saved to: {output_dir}")
    print("-------------------------------------------")

if __name__ == '__main__':
    main()