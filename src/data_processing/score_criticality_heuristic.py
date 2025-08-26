# src/data_processing/score_criticality_heuristic.py

# This script should be run with the main development environment activated.
# conda activate longtail-rl
# python -m src.data_processing.score_criticality_heuristic

import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil
from scipy.spatial import cKDTree

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.utils import geometry

# --- Global Configuration for Worker Processes ---
CONFIG = None

def init_worker(config_path: str):
    """Initializer for each worker process to load the config."""
    global CONFIG
    CONFIG = load_config(config_path)

# --- Heuristic Calculation Functions (Advanced) ---

def calculate_kinematic_volatility_scores(sdc_route: np.ndarray) -> np.ndarray:
    """
    Pillar 1: Ego-Centric Volatility
    Calculates a score based on the SDC's jerk (change in acceleration) and
    yaw acceleration (change in turn rate). High values indicate a rapid
    change in the SDC's plan.
    """
    # Timestep duration
    dt = 0.1
    
    # Calculate velocities, accelerations, and jerk
    velocities = sdc_route[:, 7:9]  # vx, vy
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = np.diff(speeds, prepend=speeds[0]) / dt
    jerks = np.diff(accelerations, prepend=accelerations[0]) / dt
    
    # Calculate yaws, yaw rates, and yaw acceleration
    yaws = sdc_route[:, 6]
    yaw_rates = np.diff(np.unwrap(yaws), prepend=yaws[0]) / dt
    yaw_accelerations = np.diff(yaw_rates, prepend=yaw_rates[0]) / dt
    
    # Normalize the absolute values. We use a fixed, physically-motivated scale
    # instead of percentiles to ensure scores are comparable across scenarios.
    # Jerk > 8 m/s^3 is very high (emergency brake).
    # Yaw Accel > 3 rad/s^2 is very high (sudden swerve).
    jerk_score = np.clip(np.abs(jerks) / 2.5, 0, 1)
    yaw_accel_score = np.clip(np.abs(yaw_accelerations) / 1.5, 0, 1)
    
    # Combine the scores
    volatility_score = np.maximum(jerk_score, yaw_accel_score)
    return volatility_score

def calculate_interaction_scores_vectorized(
    all_trajectories: np.ndarray, 
    valid_mask: np.ndarray, 
    sdc_track_index: int
) -> np.ndarray:
    """
    Pillar 2: Scene Interaction Complexity
    Calculates a score based on how directly other agents are moving towards the SDC.
    A large negative dot product of relative position and relative velocity indicates
    a high-risk "converging" interaction, even if far away.
    """
    num_agents, num_timesteps, _ = all_trajectories.shape
    sdc_traj = all_trajectories[sdc_track_index]
    sdc_traj_expanded = np.expand_dims(sdc_traj, axis=0)
    
    # Calculate relative kinematics for all agents and timesteps at once
    relative_pos = all_trajectories[:, :, :2] - sdc_traj_expanded[:, :, :2]
    relative_vel = all_trajectories[:, :, 7:9] - sdc_traj_expanded[:, :, 7:9]
    
    # Calculate the dot product for each agent at each timestep
    # A negative value means they are moving towards each other.
    # We use einsum for a fast, batch dot product.
    # 'ati,ati->at' means for each agent (a) and timestep (t), multiply and sum over the dimension (i=2).
    dot_products = np.einsum('ati,ati->at', relative_pos, relative_vel)
    
    # We only care about converging agents (negative dot product), so we take the negative.
    # A higher score now means higher convergence risk.
    convergence_risk = -np.clip(dot_products, None, 0)
    
    # Mask out the SDC and invalid agents
    valid_other_agent_mask = valid_mask.copy()
    valid_other_agent_mask[sdc_track_index, :] = False
    convergence_risk[~valid_other_agent_mask] = 0
    
    # The interaction score for a timestep is the MAXIMUM risk from any single other agent.
    interaction_scores_raw = np.max(convergence_risk, axis=0)
    
    # Normalize the score. A value of 200 is a high-risk interaction
    # (e.g., 20 m/s relative speed at 10m distance).
    interaction_scores = np.clip(interaction_scores_raw / 1000.0, 0, 1)
    
    return interaction_scores

# ### NEW FUNCTION: Trajectory Off-Road Proximity ###
def calculate_off_road_proximity_scores(
    sdc_route: np.ndarray,
    map_polylines: list,
    map_polyline_types: list
) -> np.ndarray:
    """
    Pillar 1b: Trajectory Off-Road Score
    Calculates a score based on the SDC's proximity to non-drivable map
    features like road edges and medians.
    """
    num_timesteps = sdc_route.shape[0]
    off_road_scores = np.zeros(num_timesteps, dtype=np.float32)

    # --- Isolate only the road boundary polylines ---
    # Parser IDs for road edges are 20 (Unknown), 21 (Boundary), 22 (Median)
    boundary_type_ids = {20, 21, 22} 
    
    boundary_polylines = [
        poly for poly, type_id in zip(map_polylines, map_polyline_types) 
        if type_id in boundary_type_ids
    ]

    if not boundary_polylines:
        return off_road_scores # Return all zeros if no boundaries in scene

    # For efficiency, create a single KD-Tree of all points from all boundary polylines
    all_boundary_points = np.vstack([p[:, :2] for p in boundary_polylines if p.shape[0] > 0])
    
    if all_boundary_points.shape[0] == 0:
        return off_road_scores

    kdtree = cKDTree(all_boundary_points)
    
    # --- Calculate Minimum Distance for Each Timestep ---
    sdc_positions = sdc_route[:, :2]
    # Query the KD-Tree to find the distance from each SDC position to the single nearest boundary point
    min_distances, _ = kdtree.query(sdc_positions, k=1)
    
    # --- Scoring ---
    # The score increases sharply as the SDC gets very close to a boundary.
    proximity_threshold = 0.75  # Getting closer than 75cm is now a max-score event
    
    # Create a mask for timesteps where the SDC is closer than the threshold
    score_mask = min_distances < proximity_threshold
    
    # The score is linear from 1 (at 0 distance) to 0 (at threshold distance)
    off_road_scores[score_mask] = 1.0 - (min_distances[score_mask] / proximity_threshold)
    
    return off_road_scores

# ### NEW FUNCTION: Lane Deviation Score ###
def calculate_lane_deviation_scores(
    sdc_route: np.ndarray,
    map_polylines: list,
    map_polyline_types: list
) -> np.ndarray:
    """
    Calculates a score based on the SDC's lateral distance to the nearest
    lane centerline. High distance indicates a lane change or non-standard driving.
    """
    num_timesteps = sdc_route.shape[0]
    lane_deviation_scores = np.zeros(num_timesteps, dtype=np.float32)

    # Filter to get only the lane centerlines
    # Parser IDs for all lane subtypes are 0, 1, 2, 3
    lane_polylines = [
        poly for poly, type_id in zip(map_polylines, map_polyline_types) 
        if type_id in {0, 1, 2, 3}
    ]

    if not lane_polylines:
        return lane_deviation_scores

    # To avoid re-implementing complex geometry, we can use a simplified but effective
    # proxy: the distance to the single nearest point on any lane centerline.
    # This is a good trade-off for speed in this large-scale processing script.
    all_lane_points = np.vstack([p[:, :2] for p in lane_polylines if p.shape[0] > 0])
    if all_lane_points.shape[0] == 0:
        return lane_deviation_scores

    kdtree = cKDTree(all_lane_points)
    
    sdc_positions = sdc_route[:, :2]
    min_distances, _ = kdtree.query(sdc_positions, k=1)
    
    # The score is proportional to the distance.
    # A distance > 2.0 meters is considered a significant deviation.
    max_dist_for_score = 1.25
    lane_deviation_scores = np.clip(min_distances / max_dist_for_score, 0, 1)

    return lane_deviation_scores

def calculate_social_density_scores(valid_mask: np.ndarray) -> np.ndarray:
    """Calculates a simple score based on the number of agents in the scene."""
    num_agents = np.sum(valid_mask, axis=0)
    # Normalize by a reasonable number of agents for a dense scene.
    density_score = np.clip(num_agents / 50.0, 0, 1)
    return density_score

# --- Main Worker Function ---
def process_shard(npz_path: str) -> bool:
    """Processes a single scenario .npz file and saves the advanced heuristic scores."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        scenario_id = data['scenario_id'].item()

        # --- Calculate each advanced heuristic score component ---
        volatility_scores = calculate_kinematic_volatility_scores(data['sdc_route'])
        
        interaction_scores = calculate_interaction_scores_vectorized(
            data['all_agent_trajectories'], data['valid_mask'], data['sdc_track_index']
        )
        
        density_scores = calculate_social_density_scores(data['valid_mask'])
        
        # ### Calculate off-road proximity score ###
        off_road_scores = calculate_off_road_proximity_scores(
            data['sdc_route'],
            list(data['map_polylines']), # Convert object array to list for iteration
            list(data['map_polyline_types'])
        )
        
        # --- NEW: Calculate lane deviation score ---
        lane_deviation_scores = calculate_lane_deviation_scores(
            data['sdc_route'],
            list(data['map_polylines']),
            list(data['map_polyline_types'])
        )
        
        # --- Combine scores with weighted sum from config ---
        weights = CONFIG['scoring']['heuristic']
        w_volatility = weights.get('weight_volatility', 0.4)   # Adjusted
        w_interaction = weights.get('weight_interaction', 0.4) # Adjusted
        w_off_road = weights.get('weight_off_road', 0.15)      # Adjusted
        w_density = weights.get('weight_density', 0.05)        # Adjusted
        w_deviation = weights.get('weight_lane_deviation', 0.2)  # New weight, default to 0 if not set

        final_scores = (w_volatility * volatility_scores + 
                        w_interaction * interaction_scores +
                        w_off_road * off_road_scores +
                        w_density * density_scores +
                        w_deviation * lane_deviation_scores)

        final_scores = np.clip(final_scores, 0, 1) + 0.01  # Ensure no zero scores to avoid log(0) issues later

        # --- Save the scores to the correct subdirectory ---
        output_subdir = os.path.basename(os.path.dirname(npz_path))
        output_dir = os.path.join(CONFIG['data']['criticality_scores_dir_v2'], 'timestep_level', 'heuristic', output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{scenario_id}.npz")
        # np.save(output_path, final_scores.astype(np.float32))
        
        # Uncomment to save all individual scores if needed to run tune_heuristic_weights.ipynb
        np.savez_compressed(
            output_path,
            volatility=volatility_scores.astype(np.float32),
            interaction=interaction_scores.astype(np.float32),
            off_road=off_road_scores.astype(np.float32),
            lane_deviation=lane_deviation_scores.astype(np.float32),
            density=density_scores.astype(np.float32)
        )
        
        
        return True
        
    except Exception:
        return False

# --- Main Orchestrator ---

def main():
    """
    Finds all .npz shards, processes them in parallel to calculate heuristic
    criticality scores, and saves the results.
    """
    config_path = os.path.join(PROJECT_ROOT, 'configs/main_config.yaml')
    global CONFIG
    CONFIG = load_config(config_path)

    # We score both training and validation sets
    npz_base_dir = CONFIG['data']['processed_npz_dir']
    all_npz_paths = glob(os.path.join(npz_base_dir, '*', '*.npz'))

    if not all_npz_paths:
        print(f"Error: No .npz files found in {npz_base_dir}.")
        return

    # Safe deletion logic for the output directory
    output_dir = os.path.join(CONFIG['data']['criticality_scores_dir_v2'], 'timestep_level', 'heuristic')
    
    if os.path.exists(output_dir):
        response = input(f"Output directory '{output_dir}' exists. Delete and restart? [y/N]: ")
        if response.lower() == 'y':
            shutil.rmtree(output_dir)
        else:
            print("Aborting.")
            return

    print(f"\nFound {len(all_npz_paths)} scenarios to score.")
    num_workers = CONFIG['data'].get('num_workers', cpu_count())
    print(f"Using {num_workers} worker processes.")

    with Pool(processes=num_workers, initializer=init_worker, initargs=(config_path,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_shard, all_npz_paths), total=len(all_npz_paths)))

    success_count = sum(results)
    print("\n-------------------------------------------")
    print("Heuristic Scoring complete!")
    print(f"Successfully processed and saved scores for {success_count} scenarios.")
    print(f"Failed scenarios: {len(results) - success_count}")
    print(f"Output saved to: {output_dir}")
    print("-------------------------------------------")

if __name__ == '__main__':
    main()