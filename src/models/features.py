"""Tabular feature extraction from raw look-back windows.

Given an array of shape ``(N, W)`` (one row per window), this module
computes a fixed-size feature vector per window and returns a 2-D matrix
of shape ``(N, F)`` suitable for sklearn classifiers.

Feature list (per window):
    mean, std, min, max,
    slope (linear trend),
    quantiles (10th, 25th, 75th, 90th),
    ewma_ratio (short / long EWMA at end of window),
    last_value,
    delta_from_mean  (last − mean)
"""

from __future__ import annotations

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Feature names in the order they appear in the output matrix.
FEATURE_NAMES: list[str] = [
    "mean",
    "std",
    "min",
    "max",
    "slope",
    "q10",
    "q25",
    "q75",
    "q90",
    "ewma_ratio",
    "last_value",
    "delta_from_mean",
]


def _ewma(x: np.ndarray, span: int) -> float:
    """Compute the EWMA at the last position of *x* with given *span*."""
    alpha = 2.0 / (span + 1)
    val = x[0]
    for i in range(1, len(x)):
        val = alpha * x[i] + (1.0 - alpha) * val
    return float(val)


def extract_features(X: np.ndarray) -> np.ndarray:
    """Extract tabular features from raw look-back windows.

    Parameters
    ----------
    X:
        Raw windows, shape ``(N, W)``.

    Returns
    -------
    np.ndarray, shape ``(N, F)``
        Feature matrix where ``F = len(FEATURE_NAMES)``.
    """
    N, W = X.shape
    F = len(FEATURE_NAMES)
    feats = np.empty((N, F), dtype=np.float64)

    # Pre-compute time index for slope calculation
    t = np.arange(W, dtype=np.float64)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()

    # Spans for EWMA (short ≈ 10% of W, long ≈ 50% of W, at least 2)
    short_span = max(2, W // 10)
    long_span = max(4, W // 2)

    for i in range(N):
        row = X[i]
        mu = row.mean()
        sigma = row.std()

        # Slope via closed-form OLS: β = Σ(t−t̄)(x−x̄) / Σ(t−t̄)²
        slope = ((t - t_mean) * (row - mu)).sum() / t_var if t_var > 0 else 0.0

        ewma_short = _ewma(row, short_span)
        ewma_long = _ewma(row, long_span)
        ewma_ratio = ewma_short / ewma_long if abs(ewma_long) > 1e-8 else 1.0

        feats[i] = [
            mu,
            sigma,
            row.min(),
            row.max(),
            slope,
            np.percentile(row, 10),
            np.percentile(row, 25),
            np.percentile(row, 75),
            np.percentile(row, 90),
            ewma_ratio,
            row[-1],
            row[-1] - mu,
        ]

    logger.info(
        "Extracted %d features from %d windows (W=%d).", F, N, W
    )
    return feats
