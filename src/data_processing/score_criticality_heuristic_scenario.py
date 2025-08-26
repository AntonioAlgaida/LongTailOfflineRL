# src/data_processing/score_criticality_scenario.py

# Calculates a single difficulty score for each full scenario.
# Run AFTER score_criticality_heuristic.py (the timestep version).
# To run:
# conda activate longtail-rl
# python -m src.data_processing.score_criticality_heuristic_scenario

import os
import sys
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.utils import geometry

# --- Global CONFIG for workers ---
CONFIG = None

def init_worker(config_path: str):
    global CONFIG
    CONFIG = load_config(config_path)

def process_scenario_shard(score_npz_path: str) -> tuple:
    """
    Processes one .npz file of pre-computed timestep scores and aggregates them
    into a single scalar score for the entire scenario.
    """
    try:
        data = np.load(score_npz_path)
        scenario_id = os.path.splitext(os.path.basename(score_npz_path))[0]

        # --- Aggregate Heuristics using functions sensitive to peaks ---
        
        # For volatility and off-road proximity, the SINGLE WORST moment often defines
        # the scenario's difficulty. We use the 99th percentile as a robust max.
        volatility_agg = np.percentile(data['volatility'], 99)
        off_road_agg = np.percentile(data['off_road'], 99)
        interaction_agg = np.percentile(data['interaction'], 99) # UPDATED
        
        # For lane deviation, a standard deviation captures the variability in how much the SDC deviates from its lane.
        # This solve the scenarios where the SDC might have a deviations but is generally stable (like stopped).
        lane_deviation_agg = np.std(data['lane_deviation'])
        
        # For density, the average is also a good measure of overall crowdedness.
        density_agg = np.mean(data['density'])

        # --- Combine aggregated scores ---
        # NOTE: The weights for scenario-level scores might be different from timestep-level.
        # For now, we can use the same relative importance from the config.
        # This is something you could tune later if desired.
        weights = CONFIG['scoring']['heuristic']
        final_scenario_score = (
            weights['weight_volatility'] * volatility_agg +
            weights['weight_interaction'] * interaction_agg +
            weights['weight_off_road'] * off_road_agg +
            weights['weight_lane_deviation'] * lane_deviation_agg +
            weights['weight_density'] * density_agg
        )
        
        return scenario_id, final_scenario_score

    except Exception:
        return None, None

def main():
    config_path = os.path.join(PROJECT_ROOT, 'configs/main_config.yaml')
    global CONFIG
    CONFIG = load_config(config_path)

    scores_base_dir = os.path.join(CONFIG['data']['criticality_scores_dir_v2'], 'timestep_level', 'heuristic')
    all_score_paths = glob(os.path.join(scores_base_dir, '*', '*.npz'))

    if not all_score_paths:
        print(f"Error: No timestep-level heuristic score files found in {scores_base_dir}.")
        print("Please run score_criticality_heuristic.py first.")
        return

    print(f"\nFound {len(all_score_paths)} scenarios to aggregate scores for.")
    num_workers = CONFIG['data'].get('num_workers', cpu_count())
    print(f"Using {num_workers} worker processes.")

    with Pool(processes=num_workers, initializer=init_worker, initargs=(config_path,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_scenario_shard, all_score_paths), total=len(all_score_paths)))

    # --- Save the final dictionary of scores ---
    scenario_scores = {scenario_id: float(score) for scenario_id, score in results if scenario_id is not None}
    
    # Save to the scenario_level directory
    output_dir = os.path.join(CONFIG['data']['criticality_scores_dir_v2'], 'scenario_level')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'heuristic_scenario_scores.pt')
    
    torch.save(scenario_scores, output_path)
    
    print("\n-------------------------------------------")
    print("Scenario-level heuristic scoring complete!")
    print(f"Successfully processed and saved scores for {len(scenario_scores)} scenarios.")
    print(f"Output saved to: {output_path}")
    print("-------------------------------------------")

if __name__ == '__main__':
    main()