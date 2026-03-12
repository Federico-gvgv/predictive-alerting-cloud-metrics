"""Sliding-window extraction with future-horizon binary labelling.

For each anchor time-step *t*:

* **X_t** = ``value[t - W + 1 : t + 1]``  (look-back window of length *W*)
* **y_t** = ``1`` if **any** ``is_incident`` in ``(t, t + H]``, else ``0``

The function also returns the anchor timestamps so that evaluation can
align predictions back to the original time axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_windows(
    df: pd.DataFrame,
    W: int,
    H: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Extract sliding windows with binary future-incident labels.

    Parameters
    ----------
    df:
        DataFrame with columns ``["timestamp", "value", "is_incident"]``,
        sorted by timestamp.
    W:
        Look-back window size (number of time-steps).
    H:
        Forecast horizon (number of future time-steps to scan for incidents).
    stride:
        Step between consecutive anchor positions (default 1).

    Returns
    -------
    X : np.ndarray, shape ``(N, W)``
        Feature windows (look-back values).
    y : np.ndarray, shape ``(N,)``
        Binary labels – ``1`` if any incident occurs in ``(t, t+H]``.
    timestamps : pd.DatetimeIndex
        Anchor timestamp for each window (the "current" time-step *t*).

    Raises
    ------
    ValueError
        If the series is too short to form even one window (``len < W + H``).
    """
    n = len(df)
    if n < W + H:
        raise ValueError(
            f"Series length ({n}) is shorter than W + H = {W + H}. "
            "Cannot form any windows."
        )

    values = df["value"].to_numpy(dtype=np.float64)
    incidents = df["is_incident"].to_numpy(dtype=bool)
    ts = df["timestamp"].values  # numpy datetime64

    # Valid anchor indices: t in [W-1, n-H-1] so that
    #   look-back  = values[t-W+1 : t+1]  has length W
    #   horizon    = incidents[t+1 : t+H+1]  has length H
    anchors = np.arange(W - 1, n - H, stride)

    X = np.empty((len(anchors), W), dtype=np.float64)
    y = np.empty(len(anchors), dtype=np.int64)
    anchor_ts = np.empty(len(anchors), dtype="datetime64[ns]")

    for i, t in enumerate(anchors):
        X[i] = values[t - W + 1 : t + 1]
        y[i] = int(incidents[t + 1 : t + H + 1].any())
        anchor_ts[i] = ts[t]

    timestamps = pd.DatetimeIndex(anchor_ts)

    n_pos = int(y.sum())
    logger.info(
        "Windows created – %d total, %d positive (%.2f%%), W=%d, H=%d, stride=%d.",
        len(y),
        n_pos,
        (n_pos / len(y) * 100) if len(y) else 0.0,
        W,
        H,
        stride,
    )
    return X, y, timestamps
