"""Threshold selection and cooldown logic for alert deduplication.

Cooldown
--------
After an alert is emitted, suppress further alerts for the next *N*
time-steps to avoid flooding operators with duplicate notifications.

Threshold selection
-------------------
Sweep candidate thresholds on the **validation** set, picking the one
that reaches a target event recall while minimising false positives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import event_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


def apply_cooldown(alerts: np.ndarray, cooldown: int) -> np.ndarray:
    """Suppress repeated alerts within a cooldown window.

    Parameters
    ----------
    alerts:
        Boolean array where ``True`` = alert emitted.
    cooldown:
        Minimum number of steps between consecutive alerts.

    Returns
    -------
    np.ndarray
        Boolean array with suppressed alerts removed.
    """
    out = np.zeros_like(alerts, dtype=bool)
    last_alert = -cooldown - 1  # ensure first alert can fire
    for i in range(len(alerts)):
        if alerts[i] and (i - last_alert) > cooldown:
            out[i] = True
            last_alert = i
    return out


def _run_threshold(
    scores: np.ndarray,
    timestamps: pd.DatetimeIndex,
    series_ids: np.ndarray,
    incident_intervals_dict: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
    threshold: float,
    cooldown: int,
    total_steps: int,
    max_lead_steps: int | None = None,
    freq_seconds: float | None = None,
) -> dict:
    """Evaluate a single threshold and return event metrics."""
    raw_alerts = scores >= threshold
    alert_times_dict = {}
    for sid in np.unique(series_ids):
        mask = (series_ids == sid)
        sid_alerts = raw_alerts[mask]
        if cooldown > 0:
            sid_alerts = apply_cooldown(sid_alerts, cooldown)
        sid_ts = timestamps[mask]
        alert_times_dict[sid] = [sid_ts[i] for i in range(len(sid_alerts)) if sid_alerts[i]]

    return event_metrics(
        alert_times_dict, incident_intervals_dict, total_steps,
        max_lead_steps=max_lead_steps, freq_seconds=freq_seconds,
    )


def select_threshold(
    scores: np.ndarray,
    timestamps: pd.DatetimeIndex,
    series_ids: np.ndarray,
    incident_intervals_dict: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
    cooldown: int = 10,
    total_steps: int = 0,
    target_recall: float = 0.8,
    n_candidates: int = 100,
    max_lead_steps: int | None = None,
    freq_seconds: float | None = None,
) -> tuple[float, list[dict]]:
    """Sweep thresholds and pick the one targeting *target_recall*.

    Strategy: among all thresholds that achieve ≥ ``target_recall``
    event recall, pick the one with the lowest FP count.  If none
    reaches the target, pick the one with the highest event recall.

    Parameters
    ----------
    scores:
        Risk scores in [0, 1], shape ``(N,)``.
    timestamps:
        Anchor timestamps, aligned with *scores*.
    incident_intervals:
        Ground-truth incident intervals.
    cooldown:
        Alert cooldown in steps.
    total_steps:
        Total steps (for FP normalisation).
    target_recall:
        Desired event recall level.
    n_candidates:
        Number of thresholds to try.

    Returns
    -------
    best_threshold:
        Selected threshold.
    sweep:
        List of dicts with ``threshold``, ``event_recall``, ``fp_per_10k``
        for each candidate.
    """
    candidates = np.linspace(0.01, 0.99, n_candidates)
    sweep: list[dict] = []

    for th in candidates:
        result = _run_threshold(
            scores, timestamps, series_ids, incident_intervals_dict, th, cooldown, total_steps,
            max_lead_steps=max_lead_steps, freq_seconds=freq_seconds,
        )
        sweep.append({
            "threshold": float(th),
            "event_recall": result["event_recall"],
            "fp_per_10k": result["fp_per_10k"],
            "fp_count": result["fp_count"],
            "n_detected": result["n_detected"],
        })

    # Filter candidates that meet target recall (treat NaN as 0)
    def _recall(s: dict) -> float:
        r = s["event_recall"]
        return r if r == r else 0.0  # NaN-safe

    meeting = [s for s in sweep if _recall(s) >= target_recall]

    if meeting:
        best = min(meeting, key=lambda s: s["fp_count"])
        logger.info(
            "Threshold %.3f meets target recall %.0f%% (event_recall=%.2f, fp/10k=%.1f).",
            best["threshold"],
            target_recall * 100,
            best["event_recall"],
            best["fp_per_10k"],
        )
    else:
        best = max(sweep, key=lambda s: _recall(s))
        logger.warning(
            "No threshold reaches %.0f%% event recall. "
            "Best: threshold=%.3f (event_recall=%.2f, fp/10k=%.1f).",
            target_recall * 100,
            best["threshold"],
            best["event_recall"],
            best["fp_per_10k"],
        )

    return best["threshold"], sweep
