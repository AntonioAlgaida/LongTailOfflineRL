# tests/test_cql_agent.py

# To run:
# conda activate longtail-rl
# python -m tests.test_cql_agent

import os
import sys
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.rl.cql_agent import CQLAgent
from src.rl.networks import StateReconstructor

def test_cql_agent():
    """
    Unit test for the CQLAgent. Verifies initialization, update step,
    parameter changes, and save/load functionality.
    """
    print("--- Running Test for CQLAgent ---")
    
    # --- 1. Setup ---
    config = load_config()
    device = torch.device("cpu") # Force CPU for this test for reproducibility
    print(f"Testing on device: {device}")

    # Calculate the expected state dimension
    reconstructor = StateReconstructor(config)
    state_dim = reconstructor.end_tl
    action_dim = config['action_space']['dim']
    batch_size = 4

    # --- 2. Create Synthetic Data Batch ---
    dummy_batch = {
        'states': torch.randn(batch_size, state_dim),
        'actions': torch.randn(batch_size, action_dim),
        'rewards': torch.rand(batch_size, 1),
        'next_states': torch.randn(batch_size, state_dim),
        'dones': torch.randint(0, 2, (batch_size, 1)).float()
    }
    print(f"\nCreated a synthetic data batch of size {batch_size}.")

    # --- 3. Test Agent Instantiation and Update Step ---
    print("\n--- Testing Instantiation and a Single Update Step ---")
    try:
        agent = CQLAgent(config, device)
        
        # Store a copy of an initial critic parameter
        initial_critic_param = agent.critic.q1[0].weight.data.clone()
        
        # Run a single update step
        losses = agent.update(dummy_batch)
        
        print("✅ agent.update() executed without crashing.")
        print("Returned losses:", losses)
        
        # Check that losses are valid numbers
        for key, value in losses.items():
            assert np.isfinite(value), f"Loss '{key}' is not a finite number: {value}"
        print("✅ All returned losses are valid finite numbers.")
        
        # Check that critic parameters have been updated
        updated_critic_param = agent.critic.q1[0].weight.data
        assert not torch.equal(initial_critic_param, updated_critic_param), \
            "Critic parameters were not updated after a gradient step."
        print("✅ Critic parameters were updated.")
        
        # Check that target networks have been softly updated (i.e., they are not equal, but close)
        initial_target_param = agent.critic_target.q1[0].weight.data.clone()
        # Run another update to trigger another soft update
        agent.update(dummy_batch) 
        updated_target_param = agent.critic_target.q1[0].weight.data
        
        assert not torch.equal(initial_target_param, updated_target_param), \
            "Target network parameters did not change after soft update."
        # A soft-updated param should not be equal to the main network's param
        assert not torch.equal(agent.critic.q1[0].weight.data, updated_target_param), \
            "Target network was hard-updated, not soft-updated."
        print("✅ Target network was correctly soft-updated.")
        
    except Exception as e:
        print(f"❌ Test failed during update step: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Test Save and Load Functionality ---
    print("\n--- Testing Save and Load ---")
    try:
        # Create a temporary directory for saving models
        temp_dir = "tests/temp_models"
        os.makedirs(temp_dir, exist_ok=True)
        filename = "test_agent"
        
        # Save the agent
        agent.save(temp_dir, filename)
        print(f"Agent saved to '{temp_dir}'.")
        
        # Create a new agent instance
        new_agent = CQLAgent(config, device)
        
        # Verify that the new agent's parameters are different
        assert not torch.equal(agent.actor.policy_head[0].weight.data, 
                               new_agent.actor.policy_head[0].weight.data), \
            "New agent has identical parameters before loading."
            
        # Load the saved state into the new agent
        new_agent.load(temp_dir, filename)
        print("Agent loaded into new instance.")
        
        # Verify that the parameters are now identical
        assert torch.equal(agent.actor.policy_head[0].weight.data, 
                           new_agent.actor.policy_head[0].weight.data), \
            "Actor parameters do not match after loading."
        assert torch.equal(agent.critic.q1[0].weight.data, 
                           new_agent.critic.q1[0].weight.data), \
            "Critic parameters do not match after loading."
        print("✅ Loaded agent parameters correctly match the saved agent.")
        
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"❌ Test failed during save/load: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n--- All checks passed! ---")
    print("\n✅✅✅ CQLAgent is working correctly! ✅✅✅")

if __name__ == '__main__':
    test_cql_agent()