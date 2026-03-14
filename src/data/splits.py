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


SplitDict = dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]]


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    series_ids: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitDict:
    """Split windowed data into contiguous train / val / test sets per series.

    Parameters
    ----------
    X:
        Feature array, shape ``(N, W)``.
    y:
        Label array, shape ``(N,)``.
    timestamps:
        Anchor timestamps, length ``N``.
    series_ids:
        Series IDs array, length ``N``.
    train_ratio, val_ratio, test_ratio:
        Proportions for each split (should sum to 1.0).

    Returns
    -------
    dict
        ``{"train": (X, y, ts, sid), "val": (X, y, ts, sid), "test": (X, y, ts, sid)}``.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})."
        )

    splits_dict = {"train": [], "val": [], "test": []}

    for sid in np.unique(series_ids):
        mask = (series_ids == sid)
        X_s, y_s, ts_s, sid_s = X[mask], y[mask], timestamps[mask], series_ids[mask]
        n = len(X_s)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Guard against degenerate splits
        if train_end == 0 or val_end == train_end or val_end == n:
            logger.warning("Series %s split ratios produce an empty partition for N=%d.", sid, n)
            
        splits_dict["train"].append((X_s[:train_end], y_s[:train_end], ts_s[:train_end], sid_s[:train_end]))
        splits_dict["val"].append((X_s[train_end:val_end], y_s[train_end:val_end], ts_s[train_end:val_end], sid_s[train_end:val_end]))
        splits_dict["test"].append((X_s[val_end:], y_s[val_end:], ts_s[val_end:], sid_s[val_end:]))

    res = {}
    for name in ["train", "val", "test"]:
        part = splits_dict[name]
        if part and len(part[0][0]) > 0:
            res_X = np.vstack([p[0] for p in part if len(p[0]) > 0])
            res_y = np.concatenate([p[1] for p in part])
            res_ts = pd.DatetimeIndex(np.concatenate([p[2].values for p in part]))
            res_sid = np.concatenate([p[3] for p in part])
        else:
            res_X = np.empty((0, X.shape[1]))
            res_y = np.empty(0, dtype=np.int64)
            res_ts = pd.DatetimeIndex([])
            res_sid = np.empty(0, dtype=object)
            
        res[name] = (res_X, res_y, res_ts, res_sid)

    for name, (sx, sy, st, ssid) in res.items():
        n_pos = int(sy.sum()) if len(sy) else 0
        logger.info(
            "  %-5s split: %6d windows, %5d positive (%.2f%%)",
            name,
            len(sy),
            n_pos,
            (n_pos / len(sy) * 100) if len(sy) else 0.0,
        )

    return res
