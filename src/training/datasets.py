"""PyTorch Dataset wrapper for windowed time-series data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Wraps NumPy window arrays as a PyTorch Dataset.

    Each sample is a ``(x, y)`` tuple where:

    * ``x`` has shape ``(1, W)`` — single-channel look-back window.
    * ``y`` is a scalar float label (0.0 or 1.0).

    Parameters
    ----------
    X:
        Feature windows, shape ``(N, W)``.
    y:
        Binary labels, shape ``(N,)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Add channel dimension: (W,) → (1, W)
        return self.X[idx].unsqueeze(0), self.y[idx]
