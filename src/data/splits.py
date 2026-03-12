"""Time-based contiguous train / val / test split.

Windows are split **in temporal order** – no shuffling – so that the
training set always precedes the validation set, which always precedes
the test set.  This prevents any data leakage across splits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


SplitDict = dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitDict:
    """Split windowed data into contiguous train / val / test sets.

    Parameters
    ----------
    X:
        Feature array, shape ``(N, W)``.
    y:
        Label array, shape ``(N,)``.
    timestamps:
        Anchor timestamps, length ``N``.
    train_ratio, val_ratio, test_ratio:
        Proportions for each split (should sum to 1.0).

    Returns
    -------
    dict
        ``{"train": (X, y, ts), "val": (X, y, ts), "test": (X, y, ts)}``.

    Raises
    ------
    ValueError
        If ratios do not sum to approximately 1 or any split would be empty.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})."
        )

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Guard against degenerate splits
    if train_end == 0 or val_end == train_end or val_end == n:
        raise ValueError(
            f"Split ratios produce an empty partition for N={n}. "
            "Increase the dataset size or adjust ratios."
        )

    splits: SplitDict = {
        "train": (X[:train_end], y[:train_end], timestamps[:train_end]),
        "val": (X[train_end:val_end], y[train_end:val_end], timestamps[train_end:val_end]),
        "test": (X[val_end:], y[val_end:], timestamps[val_end:]),
    }

    for name, (sx, sy, st) in splits.items():
        n_pos = int(sy.sum())
        logger.info(
            "  %-5s split: %6d windows, %5d positive (%.2f%%), "
            "time range %s → %s",
            name,
            len(sy),
            n_pos,
            (n_pos / len(sy) * 100) if len(sy) else 0.0,
            st[0] if len(st) else "N/A",
            st[-1] if len(st) else "N/A",
        )

    return splits
