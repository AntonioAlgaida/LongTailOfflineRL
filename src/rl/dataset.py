# In src/rl/dataset.py

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from glob import glob
from tqdm import tqdm
import multiprocessing
import random
from typing import List, Dict, Tuple
import traceback

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# --- NEW Top-Level Worker for Weight Alignment ---
# Global variables for the new worker
ALIGN_WORKER_CONFIG = None
ALIGN_WORKER_FILE_PATHS = None
ALIGN_WORKER_SCORES_DICT = None

def init_align_worker(config, file_paths, scores_dict):
    """Initializes globals for the alignment worker processes."""
    global ALIGN_WORKER_CONFIG, ALIGN_WORKER_FILE_PATHS, ALIGN_WORKER_SCORES_DICT
    ALIGN_WORKER_CONFIG = config
    ALIGN_WORKER_FILE_PATHS = file_paths
    ALIGN_WORKER_SCORES_DICT = scores_dict

def align_weights_worker(indexed_chunk: List[Tuple[int, int, int]]) -> List[Tuple[int, float]]:
    """
    (Map Step) Processes a chunk of the master_index.
    
    Args:
        indexed_chunk: A list of (global_index, file_index, sample_index) tuples.

    Returns:
        A list of (global_index, final_weight) tuples.
    """
    weights_cfg = ALIGN_WORKER_CONFIG['scoring']['heuristic'] # Assuming heuristic for now
    score_type = 'heuristic' # This needs to be passed or inferred
    
    worker_weights = []
    featurized_cache = {'file_index': -1, 'data': None}

    for global_idx, file_idx, sample_idx in indexed_chunk:
        path = ALIGN_WORKER_FILE_PATHS[file_idx]
        scenario_id = os.path.basename(path).split('.')[0]
        
        final_score = 1.0 # Default weight

        if scenario_id in ALIGN_WORKER_SCORES_DICT:
            if file_idx != featurized_cache['file_index']:
                featurized_cache['file_index'] = file_idx
                featurized_cache['data'] = torch.load(path, weights_only=True)
            featurized_data = featurized_cache['data']

            try:
                original_timestep = featurized_data['timesteps'][sample_idx].item()
                score_components = ALIGN_WORKER_SCORES_DICT[scenario_id]
                
                if original_timestep < len(score_components['volatility']): # Check bounds
                    # Heuristic Score Combination
                    if score_type == 'heuristic':
                        combined_score = (
                            weights_cfg['weight_volatility'] * score_components['volatility'][original_timestep] +
                            weights_cfg['weight_interaction'] * score_components['interaction'][original_timestep] +
                            weights_cfg['weight_off_road'] * score_components['off_road'][original_timestep] +
                            weights_cfg['weight_lane_deviation'] * score_components['lane_deviation'][original_timestep] +
                            weights_cfg['weight_density'] * score_components['density'][original_timestep]
                        )
                    # Ensemble and Action Rarity (they only have one component)
                    else:
                        combined_score = score_components[score_type][original_timestep]
                    
                    # Apply the final clipping and epsilon
                    final_score  = float(np.clip(combined_score, 0, 1) + 0.01)

            except (IndexError, KeyError):
                pass # Keep default weight if something fails

        worker_weights.append((global_idx, float(final_score)))
        
    return worker_weights

# --- Worker for Parallel Indexing (for TimestepDataset) ---
def _index_worker(file_path: str) -> List[Tuple[str, int]]:
    """
    Helper to get the indices of all VALID, CONTIGUOUS transitions.
    """
    valid_transitions = []
    try:
        data = torch.load(file_path, weights_only=True)
        states = data['states']
        timesteps = data['timesteps']
        
        # A transition from index i to i+1 is valid only if the original
        # timesteps are consecutive.
        for i in range(states.shape[0] - 1):
            if timesteps[i+1] == timesteps[i] + 1:
                valid_transitions.append(i) # Store the starting index of the valid transition
        
        # We need to return the file_path and the list of valid indices
        return file_path, valid_transitions
    except Exception:
        return file_path, []

# --- NEW Top-Level Worker Function for Multiprocessing ---
def _load_score_file_worker(path: str) -> Tuple[str, Dict[str, np.ndarray]]:
    """A simple worker to load a single .npz score file."""
    try:
        sid = os.path.basename(path).split('.')[0]
        data = np.load(path)
        return sid, {key: data[key] for key in data.keys()}
    except Exception:
        return None, None
    
# ==============================================================================
# --- CLASS 1: For Timestep-Level Weighting (Map-style Dataset) ---
# ==============================================================================
# --- NEW Top-Level Worker Function for Data Loading ---
def _load_featurized_file_worker(path: str) -> Dict[str, torch.Tensor]:
    """A simple worker to load a single featurized .pt file."""
    try:
        # Load the dictionary of tensors
        return torch.load(path, weights_only=False)
    except Exception:
        # Return None for corrupted files
        print(f"Error loading file {path}: {sys.exc_info()[1]}")
        return None
    
class OfflineRLTimestepDataset(Dataset):
    """
    (V3 - High-Performance In-Memory) A Map-style Dataset that pre-loads the
    entire featurized dataset into RAM for maximum training throughput.
    """
    def __init__(self, file_paths: List[str]):
        super().__init__()
        if len(file_paths) > 10000:
            print(f"Warning: Reducing dataset size from {len(file_paths)} to 10000 for performance.")
            file_paths = file_paths[:10000]
            
        self.file_paths_original = file_paths # Keep original list
        
        # --- PARALLELIZED Pre-loading of all data into RAM ---
        print(f"Pre-loading all {len(file_paths)} featurized files into RAM (in parallel)...")
        
        num_workers = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(_load_featurized_file_worker, file_paths),
                total=len(file_paths),
                desc="Loading featurized data"
            ))

        # --- Fast Reduce Step ---
        print("\nAssembling final tensors...")
        # Filter out failed loads and create a mapping from original index to data
        # This preserves the order needed for self.file_paths
        path_to_data = {path: data for path, data in zip(file_paths, results) if data is not None}
        
        # --- Prepare lists to hold data from all scenarios ---
        # For structured data, we'll build a list of dicts for states
        all_states_dicts = []
        all_actions = []
        all_timesteps = []
        self.sample_to_file_idx = []
        
        # Rebuild a clean list of file_paths that were successfully loaded
        self.file_paths = [] 
        
        # Iterate through the original file_paths to maintain a consistent order
        for i, path in enumerate(tqdm(self.file_paths_original, desc="Aggregating data")):
            if path in path_to_data:
                # data is a list of sample dicts: [{'state':{...}, 'action':...}, ...]
                data = path_to_data[path]
                num_samples = len(data)
                if num_samples > 0:
                    # --- EFFICIENT BATCH APPENDING ---
                    all_states_dicts.extend([sample['state'] for sample in data])
                    all_actions.extend([sample['action'] for sample in data])
                    all_timesteps.extend([sample['timestep'] for sample in data])
                    
                    new_file_idx = len(self.file_paths)
                    self.sample_to_file_idx.extend([new_file_idx] * num_samples)
                    self.file_paths.append(path)

        # --- 3. Final Collation into Tensors (CRITICAL PART) ---
        print("\nCollating aggregated data into final tensors...")
        
        # Collate the simple arrays first
        self.actions = torch.from_numpy(np.array(all_actions, dtype=np.float32))
        self.timesteps = torch.from_numpy(np.array(all_timesteps, dtype=np.int32))
        self.sample_to_file_idx = np.array(self.sample_to_file_idx)
        
        # Collate the structured state dictionaries key by key
        self.states = {}
        state_keys = all_states_dicts[0].keys()
        for key in tqdm(state_keys, desc="Collating state features"):
            # This is a fast list comprehension followed by a single, efficient stack
            batched_tensors = np.stack([d[key] for d in all_states_dicts], axis=0)
            self.states[key] = torch.from_numpy(batched_tensors).float()

        # --- 4. Build the master_index of valid, contiguous transitions ---
        print("Finding all valid contiguous transitions...")
        # Use torch for GPU acceleration if available
        timesteps_tensor = self.timesteps
        contiguous_mask = (timesteps_tensor[1:] == timesteps_tensor[:-1] + 1)
        
        file_boundary_mask = torch.from_numpy(self.sample_to_file_idx[1:] == self.sample_to_file_idx[:-1])
        
        # The valid start indices are where both conditions are true
        self.master_index = torch.where(contiguous_mask & file_boundary_mask)[0].numpy()

        print(f"Pre-loading complete. Found {len(self.master_index)} valid transitions from {len(self.file_paths)} files.")

    def __len__(self) -> int:
        return len(self.master_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = self.master_index[idx]
        
        # --- Assemble the structured state dictionary for state_t ---
        state_t = {key: tensor[start_idx] for key, tensor in self.states.items()}
        
        # --- Assemble the structured state dictionary for next_state_t ---
        next_state_t = {key: tensor[start_idx + 1] for key, tensor in self.states.items()}
        
        action_t = self.actions[start_idx]
        
        is_done = torch.tensor([float(start_idx + 1 >= len(self.actions) or \
                                    self.sample_to_file_idx[start_idx+1] != self.sample_to_file_idx[start_idx])])
        
        # The reward function needs to be updated to accept dicts, or we pass it
        # the components it needs.
        # For now, let's keep it simple. The agent will re-calculate it anyway.
        reward = torch.tensor([0.0], dtype=torch.float32) 
        
        return {
            'states': state_t,
            'actions': action_t,
            'rewards': reward,
            'next_states': next_state_t,
            'dones': is_done
        }

    # ### SIMPLIFIED AND CORRECTED load_and_align_weights ###
    def load_and_align_weights(self, score_dir: str, config: dict) -> torch.Tensor:
        """
        (Simplified Version) Loads pre-computed RAW score components and aligns
        them with the pre-loaded, in-memory master_index.
        """
        print(f"Loading and aligning timestep weights from: {score_dir}")
        
        # --- Step 1: Pre-load all score files in parallel (this is still a good optimization) ---
        all_score_files = glob(os.path.join(score_dir, '*', '*.npz'))
        id_to_scores_dict = {}
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap_unordered(_load_score_file_worker, all_score_files),
                                total=len(all_score_files), desc="Loading score files (parallel)"))
        for sid, data in results:
            if sid is not None: id_to_scores_dict[sid] = data
        
        # --- Step 2: Get combination weights from config ---
        score_type = os.path.basename(score_dir)
        weights_cfg = config['scoring'].get(score_type)
        if not weights_cfg:
            print(f"Warning: No weights in config for '{score_type}'. Using uniform weights.")
            return torch.ones(len(self), dtype=torch.float32)

        # --- Step 3: Align weights in a fast, sequential loop ---
        # This is fast now because all the data is already in RAM.
        final_weights = torch.ones(len(self), dtype=torch.float32)

        for i, start_idx in enumerate(tqdm(self.master_index, desc="Aligning weights")):
            file_idx = self.sample_to_file_idx[start_idx]
            path = self.file_paths[file_idx]
            scenario_id = os.path.basename(path).split('.')[0]
            
            if scenario_id in id_to_scores_dict:
                original_timestep = self.timesteps[start_idx].item()
                score_components = id_to_scores_dict[scenario_id]
                
                # Check bounds
                if original_timestep < len(score_components[list(score_components.keys())[0]]):
                    if score_type == 'heuristic':
                        combined_score = (
                            weights_cfg['weight_volatility'] * score_components['volatility'][original_timestep] +
                            weights_cfg['weight_interaction'] * score_components['interaction'][original_timestep] +
                            weights_cfg['weight_off_road'] * score_components['off_road'][original_timestep] +
                            weights_cfg['weight_lane_deviation'] * score_components['lane_deviation'][original_timestep] +
                            weights_cfg['weight_density'] * score_components['density'][original_timestep]
                        )
                    else:
                        combined_score = score_components[score_type][original_timestep]
                    
                    final_weights[i] = float(np.clip(combined_score, 0, 1) + 0.01)

        print("Weight alignment complete.")
        return final_weights

    
# ==============================================================================
# --- CLASS 2: For Scenario-Level Weighting (Iterable Dataset) ---
# ==============================================================================
class OfflineRLScenarioDataset(IterableDataset):
    """
    A high-performance IterableDataset for Offline RL that performs weighted
    sampling at the SCENARIO level.

    This dataset is used for agents like CQL-HS, CQL-ES, and CQL-ARS.
    It defines an "epoch" as a pass through a number of scenarios equal to
    the dataset size, where high-scoring scenarios are sampled more frequently.
    """
    def __init__(self, file_paths: List[str], scenario_scores: Dict[str, float]):
        super().__init__()
        self.file_paths = file_paths
        
        # --- 1. Pre-compute Scenario Sampling Probabilities ---
        print("Initializing ScenarioDataset: preparing scenario weights...")
        
        # Extract scenario IDs from file paths
        scenario_ids = [os.path.splitext(os.path.basename(p))[0] for p in self.file_paths]
        
        # Create a weight for each file path, using a default of 0.01 if a score is missing
        weights = np.array(
            [scenario_scores.get(sid, 0.01) for sid in scenario_ids], 
            dtype=np.float32
        )
        
        # Add a small epsilon to all weights to ensure every scenario has a non-zero chance
        weights += 0.01
        
        # Normalize the weights to form a probability distribution
        self.sampling_probabilities = weights / np.sum(weights)
        
        # Sanity check
        if len(self.file_paths) != len(self.sampling_probabilities):
            raise ValueError("Mismatch between number of files and probabilities.")
            
        print("Scenario weights and probabilities prepared.")
        
    def __iter__(self):
        """
        The iterator method that will be called by the DataLoader.
        """
        # Determine which subset of scenarios this worker is responsible for.
        # This is the standard pattern for multi-worker IterableDatasets.
        # --- 2. Multi-Worker Splitting Logic (Correct) ---
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_paths = self.file_paths
            worker_probs = self.sampling_probabilities
        else:
            per_worker = int(np.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.file_paths))
            worker_paths = self.file_paths[start:end]
            worker_probs = self.sampling_probabilities[start:end]
            if np.sum(worker_probs) > 0:
                worker_probs /= np.sum(worker_probs)
            else:
                worker_probs = None
            
        # --- 3. Weighted Sampling of Scenarios (Correct) ---
        num_scenarios_to_yield = len(worker_paths)
        if worker_probs is None:
            sampled_indices = np.random.choice(len(worker_paths), size=num_scenarios_to_yield, replace=True)
        else:
            sampled_indices = np.random.choice(len(worker_paths), size=num_scenarios_to_yield, replace=True, p=worker_probs)

        # --- 4. Main Data Yielding Loop (CORRECTED) ---
        for i in sampled_indices:
            scenario_path = worker_paths[i]
            try:
                # Load the list of sample dictionaries
                scenario_samples = torch.load(scenario_path, weights_only=False)
                num_samples = len(scenario_samples)
                if num_samples <= 1: continue

                # Shuffle the order of transitions within the chosen scenario
                indices = list(range(num_samples - 1)) # We can make num_samples-1 transitions
                random.shuffle(indices)
                
                for sample_idx in indices:
                    current_sample = scenario_samples[sample_idx]
                    next_sample = scenario_samples[sample_idx + 1]
                    
                    # --- Temporal Contiguity Check ---
                    if next_sample['timestep'] != current_sample['timestep'] + 1:
                        continue
                        
                    is_done = torch.tensor([float(sample_idx + 1 == num_samples - 1)])
                    
                    # Yield a dictionary with all required keys
                    yield {
                        'states': current_sample['state'],
                        'actions': current_sample['action'],
                        'rewards': torch.tensor([0.0]), # Placeholder reward, will be re-calculated in agent
                        'next_states': next_sample['state'],
                        'dones': is_done
                    }
            except Exception:
                print(f"Error loading scenario {scenario_path}: {sys.exc_info()[1]}")
                traceback.print_exc()
                continue