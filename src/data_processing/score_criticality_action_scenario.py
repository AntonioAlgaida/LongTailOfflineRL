# src/data_processing/score_criticality_action_scenario.py

# Calculates scenario-level scores based on aggregated action rarity.
# Run AFTER score_criticality_action.py.
# To run:
# conda activate longtail-rl
# python -m src.data_processing.score_criticality_action_scenario

import os
import sys
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ... (PROJECT_ROOT, imports, etc.) ...
from src.utils.config import load_config

def process_scenario_shard(score_npz_path: str) -> tuple:
    """
    Processes one .npz file of pre-computed timestep action-rarity scores
    and aggregates them into a single scalar score for the entire scenario.
    """
    try:
        data = np.load(score_npz_path)
        scenario_id = os.path.splitext(os.path.basename(score_npz_path))[0]
        
        # The data contains one key: 'action_rarity'
        timestep_scores = data['action_rarity']

        if timestep_scores.shape[0] == 0:
            return None, None

        # --- Aggregate using the 95th percentile ---
        # This captures the "peak rarity" of the actions in the scenario,
        # making it sensitive to single, highly unusual maneuvers.
        scenario_score = np.percentile(timestep_scores, 95)
        
        return scenario_id, scenario_score

    except Exception:
        return None, None

def main():
    config = load_config()

    # Read from the timestep-level action_rarity directory
    scores_base_dir = os.path.join(config['data']['criticality_scores_dir_v2'], 'timestep_level', 'action_rarity')
    all_score_paths = glob(os.path.join(scores_base_dir, '*', '*.npz'))

    if not all_score_paths:
        print(f"Error: No timestep-level action-rarity score files found in {scores_base_dir}.")
        print("Please run score_criticality_action.py first.")
        return

    print(f"\nFound {len(all_score_paths)} scenarios to aggregate action-rarity scores for.")
    num_workers = config['data'].get('num_workers', cpu_count())
    print(f"Using {num_workers} worker processes.")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_scenario_shard, all_score_paths), total=len(all_score_paths)))

    # --- Save the final dictionary of scores ---
    scenario_scores = {scenario_id: float(score) for scenario_id, score in results if scenario_id is not None}
    
    output_dir = os.path.join(config['data']['criticality_scores_dir_v2'], 'scenario_level')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'action_rarity_scenario_scores.pt')
    
    torch.save(scenario_scores, output_path)
    
    print("\n-------------------------------------------")
    print("Action-Rarity (Scenario) Scoring complete!")
    print(f"Successfully processed and saved scores for {len(scenario_scores)} scenarios.")
    print(f"Output saved to: {output_path}")
    print("-------------------------------------------")

if __name__ == '__main__':
    main()