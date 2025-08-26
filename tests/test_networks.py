# tests/test_networks.py

# To run:
# conda activate longtail-rl
# python -m tests.test_networks

import os
import sys
import torch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.rl.networks import Actor, DoubleCritic, StateReconstructor

def test_networks():
    """
    Unit test for the Actor and DoubleCritic networks.
    Verifies instantiation, forward pass, output shapes, and output ranges.
    """
    print("--- Running Test for Actor and DoubleCritic Networks ---")
    
    # --- 1. Setup ---
    config = load_config()
    
    # Check for CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Calculate the expected flattened state dimension
    reconstructor = StateReconstructor(config)
    state_dim = reconstructor.end_tl
    action_dim = config['action_space']['dim']
    batch_size = 128 # Use a small batch for testing

    # --- 2. Create Synthetic Input Data ---
    # A random flat state vector, simulating output from the DataLoader
    dummy_flat_state = torch.randn(batch_size, state_dim, device=device)
    # A random action vector
    dummy_action = torch.randn(batch_size, action_dim, device=device)
    
    print(f"\nCreated dummy input state of shape: {dummy_flat_state.shape}")
    print(f"Created dummy input action of shape: {dummy_action.shape}")

    # --- 3. Test the Actor Network ---
    print("\n--- Testing Actor ---")
    try:
        actor = Actor(config).to(device)
        actor.eval() # Set to evaluation mode
        
        with torch.no_grad():
            # Run forward pass
            pred_actions = actor(dummy_flat_state)
            
            # Check output shape
            expected_shape = (batch_size, action_dim)
            assert pred_actions.shape == expected_shape, \
                f"Actor output shape is wrong! Expected {expected_shape}, got {pred_actions.shape}"
            print(f"✅ Actor forward pass successful. Output shape: {pred_actions.shape}")
            
            # Check output range (critical for Tanh + rescale)
            action_cfg = config['action_space']
            min_accel, max_accel = action_cfg['min_acceleration'], action_cfg['max_acceleration']
            max_yaw = action_cfg['max_yaw_rate']
            
            accel_out = pred_actions[:, 0]
            yaw_out = pred_actions[:, 1]
            
            assert torch.all(accel_out >= min_accel) and torch.all(accel_out <= max_accel), \
                "Actor acceleration output is outside the defined physical limits."
            assert torch.all(yaw_out >= -max_yaw) and torch.all(yaw_out <= max_yaw), \
                "Actor yaw rate output is outside the defined physical limits."
            print("✅ Actor output values are correctly scaled and within physical limits.")
            
    except Exception as e:
        print(f"❌ Actor test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Test the DoubleCritic Network ---
    print("\n--- Testing DoubleCritic ---")
    try:
        critic = DoubleCritic(config).to(device)
        critic.eval() # Set to evaluation mode
        
        with torch.no_grad():
            # Run forward pass
            q1_preds, q2_preds = critic(dummy_flat_state, dummy_action)
            
            # Check output shapes
            expected_shape = (batch_size, 1)
            assert q1_preds.shape == expected_shape, \
                f"Critic Q1 output shape is wrong! Expected {expected_shape}, got {q1_preds.shape}"
            assert q2_preds.shape == expected_shape, \
                f"Critic Q2 output shape is wrong! Expected {expected_shape}, got {q2_preds.shape}"
            print(f"✅ Critic forward pass successful. Output shapes: Q1={q1_preds.shape}, Q2={q2_preds.shape}")

    except Exception as e:
        print(f"❌ Critic test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- All checks passed! ---")
    print("\n✅✅✅ Actor and DoubleCritic networks are working correctly! ✅✅✅")

if __name__ == '__main__':
    test_networks()