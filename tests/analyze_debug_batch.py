# analyze_debug_batch.py
# python -m tests.analyze_debug_batch

import torch
import sys
sys.path.append('.') # Add project root
from src.utils.config import load_config
from src.rl.networks import StateReconstructor

print("--- Analyzing problematic batch ---")
batch = torch.load('debug_batch.pt')
states = batch['states']

print(f"Batch states shape: {states.shape}")

# --- Check for invalid numbers in the input data ---
has_nan = torch.isnan(states).any()
has_inf = torch.isinf(states).any()

print(f"Batch contains NaN values: {has_nan}")
print(f"Batch contains Inf values: {has_inf}")

if has_nan or has_inf:
    print("❌ PROBLEM FOUND: The input data itself is corrupted.")
    # Find the exact location
    invalid_locs = torch.where(torch.isnan(states) | torch.isinf(states))
    print("Locations (batch_idx, feature_idx):", invalid_locs)
else:
    print("✅ Input data in the batch is clean (no NaNs or Infs).")
    print("This strongly suggests the problem is exploding weights in the model.")