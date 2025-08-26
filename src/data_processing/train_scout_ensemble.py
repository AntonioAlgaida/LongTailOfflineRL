# src/data_processing/train_scout_ensemble.py

# This script trains a K-fold ensemble of simple Behavioral Cloning models.
# The disagreement of these models will be used as a criticality score.
#
# To run:
# conda activate wwm
# python -m src.data_processing.train_scout_ensemble

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from sklearn.model_selection import KFold
from glob import glob
from tqdm import tqdm
import shutil
import datetime
import multiprocessing
from typing import List, Dict, Tuple
from torch.utils.tensorboard import SummaryWriter
import traceback
import random

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.utils.loss import WeightedMSELoss

# --- 1. Helper Function for Parallel Indexing (Updated for .pt files) ---
def _index_featurized_files_worker(file_path: str) -> Tuple[str, int]:
    """Worker function to get the number of samples in a single featurized .pt file."""
    try:
        # torch.load is efficient for this.
        # We're loading a dictionary of tensors.
        data = torch.load(file_path, weights_only=False)
        num_samples = len(data)
        return file_path, num_samples
    except Exception:
        print(f"Error processing file {file_path}: {traceback.format_exc()}")
        return file_path, 0

# --- 1. NEW High-Performance Iterable Dataset ---

class StochasticEpochBCDataset(IterableDataset):
    """
    A high-performance iterable dataset for the scout model training.
    
    - Uses a "stochastic epoch": each epoch consists of a random shuffle of
      scenarios, with a fixed number of samples (k) drawn from each.
    - This avoids the massive upfront indexing cost of a MapDataset and provides
      excellent data diversity with high throughput.
    """
    def __init__(self, file_paths: list, k_samples_per_scenario: int):
        super().__init__()
        self.file_paths = file_paths
        self.k = k_samples_per_scenario
        
        # This will be used by the worker_init_fn to slice the data
        self.worker_info = torch.utils.data.get_worker_info()
        if self.worker_info is None:
            self.start = 0
            self.end = len(self.file_paths)
        else:
            per_worker = int(np.ceil(len(self.file_paths) / float(self.worker_info.num_workers)))
            self.start = self.worker_info.id * per_worker
            self.end = min(self.start + per_worker, len(self.file_paths))

    def __iter__(self):
        # Get the subset of files for this specific worker
        worker_scenario_paths = self.file_paths[self.start:self.end]
        # Shuffle the order of scenarios for this epoch
        random.shuffle(worker_scenario_paths)
        
        for scenario_path in worker_scenario_paths:
            try:
                # --- CORRECTED LOGIC ---
                # 1. Load the list of sample dictionaries
                scenario_samples_list = torch.load(scenario_path, weights_only=False)
                
                num_samples_in_scenario = len(scenario_samples_list)
                if num_samples_in_scenario == 0:
                    continue

                # 2. Randomly select k indices from the list of samples
                sample_indices = np.random.choice(
                    num_samples_in_scenario, 
                    size=self.k, 
                    replace=(num_samples_in_scenario < self.k)
                )
                
                # 3. Yield the k selected samples
                for i in sample_indices:
                    # Get the individual sample dictionary
                    sample = scenario_samples_list[i]
                    # Yield the state dictionary and the action array
                    yield sample['state'], sample['action']
                # --- END CORRECTION ---
            except Exception:
                # Silently skip corrupted files
                continue

# This helper function is required for IterableDataset with multiple workers
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # The __init__ of our dataset handles the slicing based on worker_info
    dataset.__init__(dataset.file_paths, dataset.k)


# --- 2. NEW, SIMPLIFIED PyTorch Dataset ---
class BCDataset(Dataset):
    """
    (V5 - Optimized for .pt files) A memory-efficient PyTorch Dataset that
    reads from pre-featurized .pt files using Hierarchical Indexing.
    """
    def __init__(self, file_paths: list):
        self.file_paths = file_paths
        
        # --- Parallel Indexing (same pattern, just on .pt files) ---
        print(f"Indexing {len(file_paths)} featurized files in parallel...")
        with multiprocessing.Pool() as pool:
            results = list(tqdm(
                pool.imap(_index_featurized_files_worker, file_paths),
                total=len(file_paths),
                desc="Indexing .pt Files"
            ))
        
        self.master_index = []
        path_to_idx = {path: i for i, path in enumerate(file_paths)}
        for path, num_samples in results:
            if path in path_to_idx and num_samples > 0:
                file_idx = path_to_idx[path]
                for sample_idx in range(num_samples):
                    self.master_index.append((file_idx, sample_idx))
        
        if not self.master_index: raise RuntimeError("Master index is empty.")
        print(f"Indexing complete. Found {len(self.master_index)} total valid samples.")
        
        # The file-level cache is still beneficial to avoid re-opening files
        self.cache = {'file_index': -1, 'data': None}

    def __len__(self):
        return len(self.master_index)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.master_index[idx]
        
        if file_idx == self.cache['file_index']:
            data = self.cache['data']
        else:
            try:
                path = self.file_paths[file_idx]
                # Loading a .pt file is fast
                data = torch.load(path, weights_only=False)
                self.cache = {'file_index': file_idx, 'data': data}
            except Exception as e:
                # Fallback to prevent crashing the whole training run
                print(f"Warning: Error loading file {self.file_paths[file_idx]}: {e}")
                return self.__getitem__(0)

        # Data is already featurized and is a tensor, just retrieve it.
        # No more feature extraction or action calculation needed here!
        try:
            sample = data[sample_idx]
            # The sample is already a dictionary: {'state': dict, 'action': array, 'timestep': int}
            # We return the state dictionary and the action array.
            return sample['state'], sample['action']
        except IndexError:
            print(f"Warning: Index error at sample_idx {sample_idx} for file {self.file_paths[file_idx]}. Falling back.")
            return self.__getitem__(0)

def structured_collate_fn(batch):
    """
    Collates a batch of structured (dictionary) state samples into a single batch dictionary.
    """
    # Batch is a list of (state_dict, action_tensor) tuples
    state_dicts = [item[0] for item in batch]
    actions = [item[1] for item in batch]
    
    # Collate the actions into a single tensor
    collated_actions = torch.from_numpy(np.array(actions))
    
    # Collate the state dictionaries
    collated_states = {}
    # Get keys from the first sample's state dictionary
    keys = state_dicts[0].keys()
    for key in keys:
        # Stack the tensors for each key along a new batch dimension
        collated_states[key] = torch.from_numpy(np.array([d[key] for d in state_dicts]))
        
    return collated_states, collated_actions

# --- 3. Simple MLP Model for the Scout ---
# class ScoutBCModel(nn.Module):
#     """
#     (V2 - Robust Version) An MLP with LayerNorm, a Tanh activation for bounded
#     output, and explicit action rescaling.
#     """
#     def __init__(self, input_dim: int, output_dim: int, hidden_layers: list, config: dict):
#         super().__init__()
        
#         # --- Main MLP Head ---
#         layers = []
#         prev_dim = input_dim
#         for hidden_dim in hidden_layers:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(nn.LayerNorm(hidden_dim)) # Apply LayerNorm
#             layers.append(nn.ReLU())
#             prev_dim = hidden_dim
        
#         self.head = nn.Sequential(*layers)
        
#         # --- Final Action Head ---
#         self.action_head = nn.Sequential(
#             nn.Linear(prev_dim, output_dim),
#             nn.Tanh() # Squashes the output to the range [-1, 1]
#         )
        
#         # --- Action Scaling and Biasing ---
#         action_config = config['action_space']
#         min_accel = action_config['min_acceleration']
#         max_accel = action_config['max_acceleration']
#         max_yaw_rate = action_config['max_yaw_rate']

#         action_scale = torch.tensor([
#             (max_accel - min_accel) / 2.0,
#             (max_yaw_rate - (-max_yaw_rate)) / 2.0
#         ])
#         action_bias = torch.tensor([
#             (max_accel + min_accel) / 2.0,
#             (max_yaw_rate + (-max_yaw_rate)) / 2.0
#         ])
        
#         self.register_buffer('action_scale', action_scale)
#         self.register_buffer('action_bias', action_bias)

#     def forward(self, x):
#         # 1. Pass through the main body of the network
#         x = self.head(x)
#         # 2. Get the normalized, squashed action in range [-1, 1]
#         squashed_action = self.action_head(x)
#         # 3. Rescale and shift to the correct physical action space
#         # This is the final model output
#         return squashed_action * self.action_scale + self.action_bias

class ScoutBCModel(nn.Module):
    """
    An attention-based scout model for Behavioral Cloning.
    This architecture is identical to the main Actor network, ensuring that the
    disagreement signal is generated by a model with a similar inductive bias.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        features_cfg = config['features']
        ensemble_cfg = config['scoring']['ensemble']
        
        embed_dim = ensemble_cfg.get('embed_dim', 128)

        # Define raw feature dimensions (must match FeatureExtractor)
        ego_dim = 1
        agent_dim = 10
        map_dim = features_cfg['map_points_per_polyline'] * 2
        tl_dim = 2
        goal_dim = 2
        num_goal_points = features_cfg.get('num_goal_points', 5)

        # --- Entity Encoders ---
        self.ego_encoder = nn.Sequential(nn.Linear(ego_dim, embed_dim), nn.ReLU())
        self.agent_encoder = nn.Sequential(nn.Linear(agent_dim, embed_dim), nn.ReLU())
        self.map_encoder = nn.Sequential(nn.Linear(map_dim, embed_dim), nn.ReLU())
        self.tl_encoder = nn.Sequential(nn.Linear(tl_dim, embed_dim), nn.ReLU())
        self.goal_encoder = nn.Sequential(nn.Linear(goal_dim, embed_dim), nn.ReLU())

        # --- Cross-Attention Module ---
        num_heads = ensemble_cfg.get('num_attention_heads', 4)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attention_ln = nn.LayerNorm(embed_dim)

        # --- Final Policy Head ---
        policy_head_input_dim = embed_dim + embed_dim + (num_goal_points * embed_dim) # Input: Ego (D) + Attention Summary (D) + Flattened Goal (num_goals * D)
        hidden_layers = ensemble_cfg['scout_hidden_layers']
        
        layers = []
        prev_dim = policy_head_input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        self.policy_head_base = nn.Sequential(*layers)
        
        self.action_head = nn.Sequential(
            nn.Linear(prev_dim, 2), # accel, yaw_rate
            nn.Tanh()
        )
        
        self._initialize_action_scaling()

    def _initialize_action_scaling(self):
        action_cfg = self.cfg['action_space']
        min_accel, max_accel = action_cfg['min_acceleration'], action_cfg['max_acceleration']
        max_yaw_rate = action_cfg['max_yaw_rate']
        scale = torch.tensor([(max_accel - min_accel) / 2.0, (max_yaw_rate * 2) / 2.0])
        bias = torch.tensor([(max_accel + min_accel) / 2.0, 0.0])
        self.register_buffer('action_scale', scale)
        self.register_buffer('action_bias', bias)

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode all entities
        ego_embedding = self.ego_encoder(state['ego'])
        agent_embeddings = self.agent_encoder(state['agents'])
        map_embeddings = self.map_encoder(state['map'])
        tl_embedding = self.tl_encoder(state['traffic_lights'])
        goal_embeddings = self.goal_encoder(state['goal']) # Shape: (B, num_goals, D)

        # Prepare for attention
        query = ego_embedding.unsqueeze(1)
        # Context includes agents, map, and the single traffic light feature
        context = torch.cat([agent_embeddings, map_embeddings, tl_embedding.unsqueeze(1)], dim=1)
        
        # Build padding mask
        tl_mask = state['traffic_lights_mask'] # Should be (B, 1)
        padding_mask = torch.cat([state['agents_mask'], state['map_mask'], tl_mask], dim=1)
        
        # Apply Cross-Attention
        attention_output, _ = self.cross_attention(
            query=query, key=context, value=context,
            key_padding_mask=~(padding_mask).bool(),
        )
        attention_output = self.attention_ln(attention_output.squeeze(1))
        
        # Combine features for the final decision
        # Flatten goal embeddings to (B, num_goals * D)
        goal_flat = goal_embeddings.flatten(start_dim=1)
        
        # Final feature vector: ego's state, attention summary, and goal plan
        combined_features = torch.cat([ego_embedding, attention_output, goal_flat], dim=1)
        
        # Get action
        hidden_state = self.policy_head_base(combined_features)
        squashed_action = self.action_head(hidden_state)
        final_action = squashed_action * self.action_scale + self.action_bias
        
        return final_action
    
# --- 4. Main Training Orchestrator ---
def main():
    print("--- Training Scout Model Ensemble ---")
    config = load_config()
    ensemble_config = config['scoring']['ensemble']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join('models', 'scout_ensemble')
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Normalization Stats (now a dictionary of stats) ---
    stats = np.load(config['data']['feature_stats_path_v2'])
    mean_dict = {k.replace('_mean', ''): torch.from_numpy(v).to(device).float() for k, v in stats.items() if '_mean' in k}
    std_dict = {k.replace('_std', ''): torch.from_numpy(v).to(device).float() for k, v in stats.items() if '_std' in k}
    
    
    # TensorBoard Setup
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir_base = os.path.join('runs', f'ScoutEnsemble_{timestamp}')
    print(f"TensorBoard logs will be saved to: {log_dir_base}")

    # Data Preparation
    train_data_dir = os.path.join(config['data']['featurized_dir_v2'], 'training')
    all_pt_files = np.array(sorted(glob(os.path.join(train_data_dir, '*.pt'))))
    
    kf = KFold(n_splits=ensemble_config['num_folds'], shuffle=True, random_state=42)
    
    weights_path = os.path.join(os.path.dirname(config['data']['feature_stats_path_v2']), 'action_weights_v2.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Action weights not found at {weights_path}. "
                                "Please run compute_action_weights.py first.")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_pt_files)):
        print(f"\n--- Training Fold {fold + 1}/{ensemble_config['num_folds']} ---")
        writer = SummaryWriter(log_dir=os.path.join(log_dir_base, f'fold_{fold+1}'))
        
        train_files, val_files = all_pt_files[train_idx], all_pt_files[val_idx]
        
        k_samples = ensemble_config.get('k_samples_per_scenario', 5) # Add this to config
        train_dataset = StochasticEpochBCDataset(list(train_files), k_samples)
        
        val_dataset = BCDataset(val_files)
        
        train_loader = DataLoader(
                                    train_dataset, 
                                    batch_size=ensemble_config['scout_batch_size'], 
                                    shuffle=False, # shuffle MUST be False for IterableDataset
                                    num_workers=config['data']['num_workers'],
                                    pin_memory=True, 
                                    worker_init_fn=worker_init_fn, # CRITICAL for multi-worker iterable datasets
                                    persistent_workers=True,
                                    collate_fn=structured_collate_fn
                                )
        
        val_loader = DataLoader(val_dataset,
                                batch_size=ensemble_config['scout_batch_size'],
                                shuffle=False,
                                num_workers=config['data']['num_workers'],
                                pin_memory=True,
                                persistent_workers=True,
                                collate_fn=structured_collate_fn)
        
        sample_state, _ = val_dataset[0]
        model = ScoutBCModel(config).to(device)
                
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=float(ensemble_config['scout_learning_rate']),
                                      weight_decay=float(ensemble_config['scout_weight_decay']))
        
        loss_fn = WeightedMSELoss(weights_path).to(device)
        
        best_val_loss = float('inf')
        
        for epoch in range(ensemble_config['scout_num_epochs']):
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for i, (state_dict, actions) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Train")):
                # Move batch to device
                actions = actions.to(device)
                state_dict = {k: v.to(device).float() for k, v in state_dict.items()}
                
                # --- Normalize each component of the state dictionary ---
                for key in state_dict:
                    if not key.endswith('_mask'):
                        state_dict[key] = (state_dict[key] - mean_dict[key]) / (std_dict[key] + 1e-6)

                optimizer.zero_grad()
                pred_actions = model(state_dict)
                loss = loss_fn(pred_actions, actions)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1           
            
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for state_dict, actions in val_loader:
                    # Move batch to device
                    actions = actions.to(device)
                    state_dict = {k: v.to(device).float() for k, v in state_dict.items()}
                    
                    # --- Normalize each component of the state dictionary ---
                    for key in state_dict:
                        if not key.endswith('_mask'):
                            state_dict[key] = (state_dict[key] - mean_dict[key]) / (std_dict[key] + 1e-6)

                    pred_actions = model(state_dict)
                    val_loss += loss_fn(pred_actions, actions).item()
            
            avg_val_loss = val_loss / len(val_loader)
            # Log Performance Metrics
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Validation Loss = {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(output_dir, f'scout_model_fold_{fold+1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"  -> New best model for fold {fold+1} saved.")
        writer.close()

    print("\n--- Scout Ensemble Training Complete ---")

if __name__ == '__main__':
    main()