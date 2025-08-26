# src/data_processing/score_criticality_ensemble_scenario.py

# Calculates scenario-level criticality scores based on aggregated ensemble disagreement.
# Run AFTER train_scout_ensemble.py.
# To run:
# conda activate longtail-rl
# python -m src.data_processing.score_criticality_ensemble_scenario

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
    DEVICE = torch.device("cuda")

    # Load STRUCTURED Normalization Stats from the V2 path
    stats = np.load(CONFIG['data']['feature_stats_path_v2'])
    MEAN_DICT = {k.replace('_mean', ''): torch.from_numpy(v).to(DEVICE).float() for k, v in stats.items() if '_mean' in k}
    STD_DICT = {k.replace('_std', ''): torch.from_numpy(v).to(DEVICE).float() for k, v in stats.items() if '_std' in k}

    # Load all trained scout models
    SCOUT_MODELS = []
    model_dir = os.path.join('models', 'scout_ensemble')
    model_paths = sorted(glob(os.path.join(model_dir, '*.pth')))
    if not model_paths:
        raise FileNotFoundError(f"Scout models not found in '{model_dir}'.")
    
    for path in model_paths:
        model = ScoutBCModel(CONFIG)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        SCOUT_MODELS.append(model)

def process_shard(pt_path: str) -> tuple:
    """
    Processes one pre-featurized .pt file (list of dicts), calculates the
    aggregated ensemble disagreement score, and returns the scenario ID and score.
    """
    try:
        scenario_samples = torch.load(pt_path, weights_only=False)
        scenario_id = os.path.splitext(os.path.basename(pt_path))[0]
        
        if not scenario_samples:
            return None, None
            
        # Collate all samples from the scenario into a single batch
        state_dicts, _ = structured_collate_fn(
            [(sample['state'], sample['action']) for sample in scenario_samples]
        )
        
        # Move to device and normalize
        state_dicts = {k: v.to(DEVICE).float() for k, v in state_dicts.items()}
        for key in state_dicts:
            if not key.endswith('_mask'):
                state_dicts[key] = (state_dicts[key] - MEAN_DICT[key]) / (STD_DICT[key] + 1e-6)

        # Get predictions from all models for all timesteps
        with torch.no_grad():
            all_predictions = [model(state_dicts) for model in SCOUT_MODELS]
        
        predictions_tensor = torch.stack(all_predictions, dim=0)
        variance_per_timestep = torch.var(predictions_tensor, dim=0)
        disagreement_scores = torch.sum(variance_per_timestep, dim=1)
        
        # Aggregate using the 99th percentile for a robust "peak confusion" score
        if disagreement_scores.numel() > 0:
            final_scenario_score = torch.quantile(disagreement_scores, 0.99).item()
        else:
            final_scenario_score = 0.0
        
        return scenario_id, final_scenario_score

    except Exception:
        return None, None

def main():
    config_path = os.path.join(PROJECT_ROOT, 'configs/main_config.yaml')
    global CONFIG
    CONFIG = load_config(config_path)

    scores_base_dir = CONFIG['data']['featurized_dir_v2']
    all_pt_paths = glob(os.path.join(scores_base_dir, '*', '*.pt'))
    
    if not all_pt_paths:
        print(f"Error: No featurized .pt files found in {scores_base_dir}.")
        return

    print(f"\nFound {len(all_pt_paths)} scenarios to aggregate ensemble scores for.")
    num_workers = CONFIG['data'].get('num_workers', cpu_count())
    print(f"Using {num_workers} worker processes.")

    with Pool(processes=num_workers, initializer=init_worker, initargs=(config_path,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_shard, all_pt_paths), total=len(all_pt_paths)))

    # --- Save the final dictionary of scores ---
    scenario_scores = {scenario_id: float(score) for scenario_id, score in results if scenario_id is not None}
    
    output_dir = os.path.join(CONFIG['data']['criticality_scores_dir_v2'], 'scenario_level')
    os.makedirs(output_dir, exist_ok=True)
    # Give it a distinct name
    output_path = os.path.join(output_dir, 'ensemble_scenario_scores.pt')
    
    torch.save(scenario_scores, output_path)
    
    print("\n-------------------------------------------")
    print("Ensemble-based (Scenario) Scoring complete!")
    print(f"Successfully processed and saved scores for {len(scenario_scores)} scenarios.")
    print(f"Output saved to: {output_path}")
    print("-------------------------------------------")

if __name__ == '__main__':
    main()