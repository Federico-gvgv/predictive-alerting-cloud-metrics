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
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
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
    X_list, y_list, ts_list, sid_list = [], [], [], []
    if "series_id" not in df.columns:
        df = df.copy()
        df["series_id"] = "default"

    for sid, group in df.groupby("series_id", sort=False):
        n = len(group)
        if n < W + H:
            logger.warning("Series %s too short (%d < %d). Skipping.", sid, n, W+H)
            continue

        values = group["value"].to_numpy(dtype=np.float64)
        incidents = group["is_incident"].to_numpy(dtype=bool)
        ts = group["timestamp"].values  # numpy datetime64

        # Valid anchor indices: t in [W-1, n-H-1]
        anchors = np.arange(W - 1, n - H, stride)

        X_s = np.empty((len(anchors), W), dtype=np.float64)
        y_s = np.empty(len(anchors), dtype=np.int64)
        anchor_ts_s = np.empty(len(anchors), dtype="datetime64[ns]")

        for i, t in enumerate(anchors):
            X_s[i] = values[t - W + 1 : t + 1]
            y_s[i] = int(incidents[t + 1 : t + H + 1].any())
            anchor_ts_s[i] = ts[t]

        X_list.append(X_s)
        y_list.append(y_s)
        ts_list.append(anchor_ts_s)
        sid_list.append(np.full(len(anchors), sid, dtype=object))

    if not X_list:
        raise ValueError("No series were long enough to form windows.")

    y_cat = np.concatenate(y_list)
    n_pos = int(y_cat.sum())
    
    logger.info(
        "Windows created – %d total, %d positive (%.2f%%), W=%d, H=%d, stride=%d.",
        len(y_cat),
        n_pos,
        (n_pos / len(y_cat) * 100) if len(y_cat) else 0.0,
        W,
        H,
        stride,
    )
    return (
        np.vstack(X_list),
        y_cat,
        pd.DatetimeIndex(np.concatenate(ts_list)),
        np.concatenate(sid_list),
    )
