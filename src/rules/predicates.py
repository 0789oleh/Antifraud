import torch
import torch.nn as nn


class Predicate(nn.Module):
    """
    Indivisible predicate: checks if a feature is in a specific bin.
    Takes a soft-assignment from the discretizer (batch, n_bins)
    and returns "truth degree" (batch, 1).
    """
    def __init__(self, feature_idx: int, bin_idx: int):
        super().__init__()
        self.feature_idx = feature_idx
        self.bin_idx = bin_idx

    def forward(self, x_discrete: torch.Tensor) -> torch.Tensor:
        # x_discrete: (batch, n_features, n_bins)
        # Returning the probability that the feature falls into bin_idx
        # shape: (batch, 1)
        return x_discrete[:, self.feature_idx, self.bin_idx].unsqueeze(-1)