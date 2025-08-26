# src/data_processing/featurizer.py

# This script performs the main feature extraction. It reads the raw .npz files,
# calculates ground-truth actions using a bicycle model, extracts structured
# feature dictionaries, and saves the final samples to .pt files.

# To run:
# conda activate wwm
# python -m src.data_processing.featurizer

import os

os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import sys
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil
import jax
import jax.numpy as jnp
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.rl.feature_extractor import FeatureExtractor
from waymax import datatypes
from waymax.dynamics import InvertibleBicycleModel

# --- Global Vars for Workers ---
CONFIG = None
FEATURE_EXTRACTOR = None
BICYCLE_MODEL = None

def init_worker(config_path: str):
    """Initializer for each worker process."""
    global CONFIG, FEATURE_EXTRACTOR, BICYCLE_MODEL
    CONFIG = load_config(config_path)
    FEATURE_EXTRACTOR = FeatureExtractor(CONFIG)
    BICYCLE_MODEL = InvertibleBicycleModel()
    
    
def compute_expert_actions_with_bicycle_model(
    sdc_route_np: np.ndarray, 
    sdc_valid_mask_np: np.ndarray, 
    timestamps_np: np.ndarray, 
    model: InvertibleBicycleModel
) -> np.ndarray:
    """
    (V2 - Fully Compliant) Uses the Waymax InvertibleBicycleModel to calculate
    the ground-truth expert actions for the SDC's trajectory.
    """
    num_timesteps = sdc_route_np.shape[0]
    sdc_route_jax = jnp.asarray(sdc_route_np)
    
    # Convert timestamps from seconds (float) to microseconds (int)
    timestamps_micros = (jnp.asarray(timestamps_np, dtype=float) * 1e6).astype(jnp.int32)

    # Waymax expects a batch dimension and an agent dimension.
    # We create a dummy trajectory with shape (1, 1, num_timesteps, ...)
    trajectory = datatypes.Trajectory(
        x=sdc_route_jax[jnp.newaxis, jnp.newaxis, :, 0],
        y=sdc_route_jax[jnp.newaxis, jnp.newaxis, :, 1],
        z=sdc_route_jax[jnp.newaxis, jnp.newaxis, :, 2],
        length=sdc_route_jax[jnp.newaxis, jnp.newaxis, :, 3],
        width=sdc_route_jax[jnp.newaxis, jnp.newaxis, :, 4],
        height=sdc_route_jax[jnp.newaxis, jnp.newaxis, :, 5],
        yaw=sdc_route_jax[jnp.newaxis, jnp.newaxis, :, 6],
        vel_x=sdc_route_jax[jnp.newaxis, jnp.newaxis, :, 7],
        vel_y=sdc_route_jax[jnp.newaxis, jnp.newaxis, :, 8],
        # --- ADDED REQUIRED FIELDS ---
        valid=jnp.asarray(sdc_valid_mask_np)[jnp.newaxis, jnp.newaxis, :],
        timestamp_micros=timestamps_micros[jnp.newaxis, jnp.newaxis, :]
    )
    
    # We must provide all required fields with the correct shape: (1, 1)
    # where the dimensions are (batch_size, num_objects)
    metadata = datatypes.ObjectMetadata(
        ids=jnp.array([[-1]], dtype=jnp.int32),           # Dummy ID
        object_types=jnp.array([[1]], dtype=jnp.int32),   # Treat as Vehicle
        is_sdc=jnp.array([[True]], dtype=jnp.bool_),       # This is the SDC
        is_modeled=jnp.array([[True]], dtype=jnp.bool_),   # Assume it's modeled
        is_valid=jnp.array([[True]], dtype=jnp.bool_),     # Assume it's valid overall
        objects_of_interest=jnp.array([[False]], dtype=jnp.bool_), # Not relevant for this calculation
        is_controlled=jnp.array([[True]], dtype=jnp.bool_)  # Assume it's controlled
    )

    # JIT-compile a function to run the inverse model over all timesteps
    @jax.jit
    def run_inverse_for_all_steps(traj, meta):
        return jax.vmap(model.inverse, in_axes=(None, None, 0))(traj, meta, jnp.arange(num_timesteps - 1))

    actions_jax = run_inverse_for_all_steps(trajectory, metadata)
    
    # The output is an Action dataclass. We need to access the '.data' attribute,
    # which contains the JAX array with the [accel, steer] values.
    action_data_jax = actions_jax.data
    
    # Now we can squeeze and convert the JAX array to a NumPy array.
    actions_np = np.asarray(action_data_jax.squeeze())
    
    # The inverse model can produce NaNs for invalid transitions (e.g., at zero speed).
    # We should clean these up.
    return np.nan_to_num(actions_np)

def process_shard(npz_path: str) -> bool:
    """
    Processes one .npz file, extracts structured feature dicts and actions for
    all valid timesteps, and saves them to a new .pt file.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        scenario_id = data['scenario_id'].item()
        
        num_timesteps = len(data['timestamps'])
        sdc_idx = data['sdc_track_index'].item()
        valid_mask_sdc = data['valid_mask'][sdc_idx, :]
        sdc_route = data['sdc_route']
        timestamps = data['timestamps'] # <-- Get timestamps

        # --- 1. Calculate all actions for the scenario at once ---
        expert_actions = compute_expert_actions_with_bicycle_model(
            sdc_route, 
            valid_mask_sdc, # <-- Pass the valid mask
            timestamps,     # <-- Pass the timestamps
            BICYCLE_MODEL
        )
        
        # --- 2. Create a list of structured samples ---
        scenario_samples = []
        for t in range(num_timesteps - 1):
            if valid_mask_sdc[t] and valid_mask_sdc[t+1]:
                try:
                    # extract_features now returns a dictionary
                    state_dict = FEATURE_EXTRACTOR.extract_features(data, t)
                    
                    # The action for state t is the one that leads to state t+1
                    action = expert_actions[t]

                    # Final check for validity before saving
                    if np.isfinite(action).all(): # The extractor already checks the state
                        sample = {
                            'state': state_dict, # The dictionary of np.arrays
                            'action': action,    # The (2,) np.array for the action
                            'timestep': t
                        }
                        scenario_samples.append(sample)
                except (ValueError, IndexError):
                    print(f"⚠️ Warning: Skipping invalid timestep {t} in scenario {scenario_id} at {npz_path}.")
                    print(traceback.format_exc())
                    continue

        if not scenario_samples:
            print(f"⚠️ Warning: No valid samples found in scenario {scenario_id} at {npz_path}.")
            return False

        # --- 3. Save the list of dictionaries to a .pt file ---
        output_subdir = os.path.basename(os.path.dirname(npz_path))
        output_dir = os.path.join(CONFIG['data']['featurized_dir_v2'], output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{scenario_id}.pt")
        torch.save(scenario_samples, output_path)
        
        return True
        
    except Exception:
        print(f"❌ Error processing {npz_path}: {traceback.format_exc()}")
        return False

# In src/data_processing/featurizer.py

def main():
    """
    Main orchestrator for the featurization pipeline. Finds all raw .npz files,
    processes them in parallel to extract feature vectors, and saves them to a
    new 'featurized' directory.
    """
    config_path = os.path.join(PROJECT_ROOT, 'configs/main_config.yaml')
    global CONFIG
    CONFIG = load_config(config_path)

    # --- Configuration and Path Validation ---
    if 'featurized_dir_v2' not in CONFIG['data']:
        raise ValueError("CRITICAL: Please add `data.featurized_dir_v2` to your main_config.yaml. "
                         "For example: featurized_dir_v2: '/mnt/d/waymo_longtail_rl_data/featurized'")
    
    output_dir = CONFIG['data']['featurized_dir_v2']
    npz_base_dir = CONFIG['data']['processed_npz_dir']
    all_npz_paths = glob(os.path.join(npz_base_dir, '*', '*.npz'))

    if not all_npz_paths:
        print(f"❌ Error: No source .npz files found in '{npz_base_dir}'. Please run the parser first.")
        return

    # --- Safe Deletion / Incremental Processing Logic ---
    paths_to_process = all_npz_paths

    if os.path.exists(output_dir):
        print(f"\nOutput directory '{output_dir}' already exists.")
        response = input("Choose an action: [d]elete and start fresh, [c]ontinue and skip existing, or [a]bort? [d/c/a]: ")
        
        if response.lower() == 'd':
            print("Deleting existing output directory...")
            shutil.rmtree(output_dir)
            print("Directory deleted.")
            # We don't need to recreate it here; workers will do it.
        
        elif response.lower() == 'c':
            print("Continuing. Will skip files that have already been processed.")
            # Get a set of scenario IDs that already exist in the output directory
            existing_ids = set()
            for subdir in ['training', 'validation']:
                path = os.path.join(output_dir, subdir)
                if os.path.exists(path):
                    existing_ids.update(
                        [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.pt')]
                    )

            # Filter the list of .npz paths to only include those not yet processed
            paths_to_process = [
                p for p in all_npz_paths
                if os.path.splitext(os.path.basename(p))[0] not in existing_ids
            ]
            print(f"Found {len(existing_ids)} already featurized scenarios.")
            
        else:
            print("Aborting.")
            return
    
    if not paths_to_process:
        print("\nAll scenarios have already been featurized. Nothing to do.")
        return

    print(f"\nFound {len(paths_to_process)} new scenarios to featurize (out of {len(all_npz_paths)} total).")
    
    # --- Parallel Processing ---
    num_workers = cpu_count() - 2
    print(f"Using {num_workers} worker processes.")

    with Pool(processes=num_workers, initializer=init_worker, initargs=(config_path,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_shard, paths_to_process), total=len(paths_to_process)))

    # --- Final Summary ---
    success_count = sum(results)
    print("\n-------------------------------------------")
    print("Featurization complete!")
    print(f"Successfully processed and saved features for {success_count} scenarios.")
    print(f"Failed or empty scenarios during this run: {len(results) - success_count}")
    print(f"Output saved to: {output_dir}")
    print("-------------------------------------------")


if __name__ == '__main__':
    main()