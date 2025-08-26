# src/data_processing/validate_trajectories.py

# Scans the processed .npz dataset to check for violations of physical
# limits in the SDC trajectories.
# To run:
# conda activate longtail-rl
# python -m src.data_processing.validate_trajectories

import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config

def analyze_trajectory_shard(npz_path: str) -> dict:
    """
    Analyzes the SDC trajectory in a single scenario file for physical violations.
    Returns a dictionary with the most extreme values found.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        sdc_route = data['sdc_route']
        valid_mask = data['valid_mask'][data['sdc_track_index'], :]
        
        # Filter to only valid, contiguous segments
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 3: # Need at least 3 points for jerk
            return {}
            
        # Keep only the longest contiguous block of valid states
        blocks = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
        longest_block = max(blocks, key=len)
        if len(longest_block) < 3:
            return {}

        traj = sdc_route[longest_block]
        dt = 0.1 # Timestep

        # --- Calculate Kinematics ---
        positions = traj[:, :2]
        velocities = traj[:, 7:9]
        speeds = np.linalg.norm(velocities, axis=1)
        yaws = traj[:, 6]

        accelerations = np.diff(speeds, prepend=speeds[0]) / dt
        jerks = np.diff(accelerations, prepend=accelerations[0]) / dt
        yaw_rates = np.diff(np.unwrap(yaws), prepend=yaws[0]) / dt
        
        position_jumps = np.linalg.norm(np.diff(positions, axis=0, prepend=positions[0:1]), axis=1)

        # --- Find Maximum Absolute Values ---
        max_values = {
            'max_speed': np.max(speeds),
            'max_accel': np.max(accelerations),
            'min_accel': np.min(accelerations), # Max deceleration
            'max_jerk': np.max(np.abs(jerks)),
            'max_yaw_rate': np.max(np.abs(yaw_rates)),
            'max_pos_jump': np.max(position_jumps),
            'scenario_id': data['scenario_id'].item()
        }
        return max_values
        
    except Exception:
        return {}

def main():
    """Finds all .npz shards and processes them in parallel."""
    config = load_config()
    
    # Analyze the validation set as a representative sample
    npz_base_dir = os.path.join(config['data']['processed_npz_dir'], 'validation')
    all_npz_paths = glob(os.path.join(npz_base_dir, '*.npz'))

    if not all_npz_paths:
        print(f"Error: No .npz files found in {npz_base_dir}.")
        return

    print(f"\nAnalyzing {len(all_npz_paths)} scenario trajectories for physical limit violations...")
    num_workers = config['data'].get('num_workers', cpu_count())
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(analyze_trajectory_shard, all_npz_paths), total=len(all_npz_paths)))

    # --- Aggregate and Report Results ---
    # Filter out empty results from failed files
    results = [res for res in results if res]
    if not results:
        print("No valid scenarios were processed.")
        return
        
    print("\n--- Trajectory Analysis Complete ---")
    print("\n--- Overall Maximum Values Found Across Dataset ---")
    
    # Find the single most extreme value for each metric
    max_speed = max(res['max_speed'] for res in results)
    max_accel = max(res['max_accel'] for res in results)
    min_accel = min(res['min_accel'] for res in results)
    max_jerk = max(res['max_jerk'] for res in results)
    max_yaw_rate = max(res['max_yaw_rate'] for res in results)
    max_pos_jump = max(res['max_pos_jump'] for res in results)

    # Find the scenarios responsible for these extremes
    max_speed_scen = next(res['scenario_id'] for res in results if res['max_speed'] == max_speed)
    max_accel_scen = next(res['scenario_id'] for res in results if res['max_accel'] == max_accel)
    min_accel_scen = next(res['scenario_id'] for res in results if res['min_accel'] == min_accel)
    max_jerk_scen = next(res['scenario_id'] for res in results if res['max_jerk'] == max_jerk)
    max_yaw_rate_scen = next(res['scenario_id'] for res in results if res['max_yaw_rate'] == max_yaw_rate)
    max_pos_jump_scen = next(res['scenario_id'] for res in results if res['max_pos_jump'] == max_pos_jump)

    print(f"  - Max Speed:        {max_speed:.2f} m/s (in scenario {max_speed_scen})")
    print(f"  - Max Acceleration: {max_accel:.2f} m/s^2 (in scenario {max_accel_scen})")
    print(f"  - Max Deceleration: {min_accel:.2f} m/s^2 (in scenario {min_accel_scen})")
    print(f"  - Max Jerk:         {max_jerk:.2f} m/s^3 (in scenario {max_jerk_scen})")
    print(f"  - Max Yaw Rate:     {max_yaw_rate:.2f} rad/s (in scenario {max_yaw_rate_scen})")
    print(f"  - Max Pos. Jump:    {max_pos_jump:.2f} m/step (in scenario {max_pos_jump_scen})")

    print("\n--- Analysis ---")
    print(" - Reasonable limits: Accel [~-9.0, ~4.0], Yaw Rate [~1.0], Pos. Jump [~3.0 for 60mph].")
    print(" - High jerk or position jump values can indicate noisy or unrealistic data points.")
    print(" - This analysis helps validate the quality of the source dataset before inferring actions.")

if __name__ == '__main__':
    main()