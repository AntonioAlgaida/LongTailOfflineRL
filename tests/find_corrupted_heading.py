# tests/find_corrupted_heading.py
# python -m tests.find_corrupted_heading
import numpy as np
import os
import sys
from glob import glob
from tqdm import tqdm
from src.utils.config import load_config

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config

def find_corruption():
    config = load_config()
    data_dir = os.path.join(config['data']['processed_npz_dir'], 'validation')
    all_files = glob(os.path.join(data_dir, '*.npz'))

    print(f"Scanning {len(all_files)} files for corrupted heading values...")

    for path in tqdm(all_files):
        try:
            data = np.load(path, allow_pickle=True)
            trajectories = data['all_agent_trajectories']
            valid_mask = data['valid_mask']
            
            # Find any heading value outside the valid [-pi, pi] range
            # We use a small tolerance.
            corrupted_headings = trajectories[:, :, 6][valid_mask]
            
            if np.any(np.abs(corrupted_headings) > 3.15):
                # We found one!
                # Find the exact location
                locations = np.where(np.abs(trajectories[:, :, 6]) > 3.15)
                agent_idx = locations[0][0]
                timestep_idx = locations[1][0]
                bad_value = trajectories[agent_idx, timestep_idx, 6]

                print(f"\n\n--- CORRUPTION DETECTED! ---")
                print(f"File: {os.path.basename(path)}")
                print(f"Agent Index: {agent_idx}")
                print(f"Timestep: {timestep_idx}")
                print(f"Corrupted Heading Value: {bad_value}")
                print(f"--------------------------\n")
                return # Stop after finding the first one
        except Exception:
            continue
            
if __name__ == '__main__':
    find_corruption()