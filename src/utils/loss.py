import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    """
    A custom Mean Squared Error loss function that applies a weight to each
    sample based on the rarity of its true action.
    """
    def __init__(self, weights_path):
        super().__init__()
        print(f"Loading action weights from: {weights_path}")
        weight_data = torch.load(weights_path, weights_only=True)
        self.weights = weight_data['weights'].requires_grad_(False)
        self.accel_bins = weight_data['accel_bins'].requires_grad_(False)
        self.steer_bins = weight_data['steer_bins'].requires_grad_(False)

    def to(self, device):
        """Moves the weight tensors to the specified device."""
        self.weights = self.weights.to(device)
        self.accel_bins = self.accel_bins.to(device)
        self.steer_bins = self.steer_bins.to(device)
        return self

    def forward(self, pred_actions, true_actions):
        # --- FIX: Make tensor slices contiguous before using them ---
        # This prevents an extra data copy and silences the UserWarning.
        true_accel = true_actions[:, 0].contiguous()
        true_steer = true_actions[:, 1].contiguous()
        
        # Determine the bin index for each true action in the batch
        # Now we pass the contiguous tensors to torch.bucketize
        accel_indices = torch.bucketize(true_accel, self.accel_bins) - 1
        steer_indices = torch.bucketize(true_steer, self.steer_bins) - 1
        
        # Clamp indices to be within the valid range of the weights tensor
        accel_indices = torch.clamp(accel_indices, 0, self.weights.shape[0] - 1)
        steer_indices = torch.clamp(steer_indices, 0, self.weights.shape[1] - 1)
        
        # Look up the weight for each action in the batch
        batch_weights = self.weights[accel_indices, steer_indices]
        
        # Calculate the standard squared error
        squared_errors = (pred_actions - true_actions) ** 2
        
        # Apply the weights. The unsqueeze(1) correctly broadcasts the
        # (batch_size,) weight tensor to the (batch_size, 2) error tensor.
        weighted_squared_errors = batch_weights.unsqueeze(1) * squared_errors
        
        # Return the mean of the weighted errors
        return torch.mean(weighted_squared_errors)