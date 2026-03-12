"""Synthetic time-series generator with labelled incidents.

Produces a reproducible time series that mimics cloud-metric behaviour:
smooth trend, daily seasonality, heavy-tailed noise, regime shifts, and
spike injections.  Regime shifts and spikes are labelled as incidents.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_synthetic(cfg: dict[str, Any]) -> pd.DataFrame:
    """Generate a synthetic cloud-metric time series.

    Parameters
    ----------
    cfg:
        Full experiment config.  Relevant keys under
        ``cfg["dataset"]["synthetic"]``:

        * ``n_samples``  – number of time-steps (default 5000).
        * ``freq``       – pandas frequency string (default ``"5min"``).
        * ``n_regimes``  – number of regime-shift events (default 3).
        * ``n_spikes``   – number of spike events (default 10).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (datetime64), ``value`` (float64),
        ``is_incident`` (bool).
    """
    syn_cfg: dict[str, Any] = cfg["dataset"].get("synthetic", {})
    n_samples: int = syn_cfg.get("n_samples", 5000)
    freq: str = syn_cfg.get("freq", "5min")
    n_regimes: int = syn_cfg.get("n_regimes", 3)
    n_spikes: int = syn_cfg.get("n_spikes", 10)
    seed: int = cfg.get("training", {}).get("seed", 42)

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64)

    # ── Base components ──────────────────────────────────────────────
    trend = 50.0 + 0.002 * t  # slow upward drift
    seasonality = 10.0 * np.sin(2 * np.pi * t / 288)  # period ≈ 1 day @ 5 min
    noise = rng.standard_t(df=3, size=n_samples) * 2.0  # heavy-tailed

    value = trend + seasonality + noise
    is_incident = np.zeros(n_samples, dtype=bool)

    # ── Regime shifts ────────────────────────────────────────────────
    regime_duration = max(20, n_samples // 50)
    regime_starts = rng.choice(
        np.arange(regime_duration, n_samples - regime_duration),
        size=n_regimes,
        replace=False,
    )
    for start in regime_starts:
        end = min(start + regime_duration, n_samples)
        shift = rng.uniform(15, 30) * rng.choice([-1, 1])
        value[start:end] += shift
        is_incident[start:end] = True

    # ── Spike injections ─────────────────────────────────────────────
    spike_width = max(3, n_samples // 500)
    spike_positions = rng.choice(
        np.arange(spike_width, n_samples - spike_width),
        size=n_spikes,
        replace=False,
    )
    for pos in spike_positions:
        s = slice(pos, min(pos + spike_width, n_samples))
        value[s] += rng.uniform(20, 50) * rng.choice([-1, 1])
        is_incident[s] = True

    # ── Assemble DataFrame ───────────────────────────────────────────
    timestamps = pd.date_range(start="2024-01-01", periods=n_samples, freq=freq)
    df = pd.DataFrame(
        {"timestamp": timestamps, "value": value, "is_incident": is_incident}
    )

    logger.info(
        "Synthetic data generated – %d rows, %d incident rows (%.2f%%).",
        len(df),
        df["is_incident"].sum(),
        df["is_incident"].mean() * 100,
    )
    return df
