"""Pointwise and event-level evaluation metrics.

Pointwise
---------
Standard classification metrics computed per window (ROC-AUC, PR-AUC,
precision, recall, F1).

Event-level
-----------
Metrics that reflect real alerting quality:

* **Event recall** – fraction of incident intervals where ≥ 1 alert fires
  within ``[incident_start − H, incident_start)`` (bounded by the
  prediction horizon *H*).
* **False positives per 10 k steps** – alerts not matched to any upcoming
  incident, normalised by duration.
* **Lead time** – for detected incidents, ``incident_start − first_alert``,
  reported in *time-steps*.  Median and IQR are included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Pointwise metrics ───────────────────────────────────────────────────


def pointwise_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true:  Binary ground-truth labels, shape ``(N,)``.
    scores:  Continuous risk scores in [0, 1], shape ``(N,)``.
    threshold:  Decision threshold.

    Returns
    -------
    dict with keys: roc_auc, pr_auc, precision, recall, f1, threshold.
    """
    preds = (scores >= threshold).astype(int)
    results: dict[str, float] = {}

    if len(np.unique(y_true)) > 1:
        results["roc_auc"] = float(roc_auc_score(y_true, scores))
        results["pr_auc"] = float(average_precision_score(y_true, scores))
    else:
        results["roc_auc"] = float("nan")
        results["pr_auc"] = float("nan")

    results["precision"] = float(precision_score(y_true, preds, zero_division=0))
    results["recall"] = float(recall_score(y_true, preds, zero_division=0))
    results["f1"] = float(f1_score(y_true, preds, zero_division=0))
    results["threshold"] = threshold
    return results


# ── Incident interval extraction ────────────────────────────────────────


def extract_incident_intervals(
    df: pd.DataFrame,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Extract contiguous incident intervals from the raw dataframe.

    Parameters
    ----------
    df:
        DataFrame with ``timestamp`` and ``is_incident`` columns.
    start_ts, end_ts:
        Optional time boundaries to restrict to a specific split.

    Returns
    -------
    List of ``(interval_start, interval_end)`` tuples.
    """
    sub = df.copy()
    if start_ts is not None:
        sub = sub[sub["timestamp"] >= start_ts]
    if end_ts is not None:
        sub = sub[sub["timestamp"] <= end_ts]

    sub = sub.sort_values("timestamp").reset_index(drop=True)
    incident = sub["is_incident"].astype(bool).values
    ts = sub["timestamp"].values

    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_incident = False
    i_start = 0

    for i in range(len(incident)):
        if incident[i] and not in_incident:
            in_incident = True
            i_start = i
        elif not incident[i] and in_incident:
            in_incident = False
            intervals.append((pd.Timestamp(ts[i_start]), pd.Timestamp(ts[i - 1])))

    # Close trailing interval
    if in_incident:
        intervals.append((pd.Timestamp(ts[i_start]), pd.Timestamp(ts[len(incident) - 1])))

    return intervals


# ── Event-level metrics ─────────────────────────────────────────────────


def event_metrics(
    alert_times: list[pd.Timestamp],
    incident_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    total_steps: int,
    max_lead_steps: int | None = None,
    freq_seconds: float | None = None,
) -> dict[str, object]:
    """Compute event-level alerting metrics.

    An incident is **detected** if at least one alert fires strictly
    *before* the incident's ``start_ts`` **and** within the maximum lead
    window.  When ``max_lead_steps`` is set, the valid detection window
    for an incident starting at *t* is::

        [t − max_lead_steps * freq, t)

    Parameters
    ----------
    alert_times:
        Sorted list of timestamps where alerts were emitted.
    incident_intervals:
        List of ``(start, end)`` tuples (sorted by start).
    total_steps:
        Total number of time-steps in the evaluation window (for FP
        normalisation).
    max_lead_steps:
        Maximum lead time in time-steps.  If ``None`` any alert before
        the incident counts (legacy behaviour).
    freq_seconds:
        Sampling interval in seconds, used to convert ``max_lead_steps``
        to a timedelta.  Required when ``max_lead_steps`` is set.

    Returns
    -------
    dict with:
        event_recall, n_detected, n_incidents,
        fp_count, fp_per_10k,
        lead_times (list of lead time in steps),
        lead_time_median_steps, lead_time_iqr_steps,
    """
    n_incidents = len(incident_intervals)
    if n_incidents == 0:
        return {
            "event_recall": float("nan"),
            "n_detected": 0,
            "n_incidents": 0,
            "fp_count": len(alert_times),
            "fp_per_10k": (len(alert_times) / max(total_steps, 1)) * 10_000,
            "lead_times": [],
            "lead_time_median_steps": float("nan"),
            "lead_time_iqr_steps": float("nan"),
        }

    # Compute max lead timedelta if bounded
    if max_lead_steps is not None and freq_seconds is not None:
        max_lead_td = pd.Timedelta(seconds=max_lead_steps * freq_seconds)
    else:
        max_lead_td = None

    # For each incident, find earliest alert within valid window
    detected = [False] * n_incidents
    lead_times: list[float] = []  # in steps
    matched_alerts: set[int] = set()

    for j, (inc_start, inc_end) in enumerate(incident_intervals):
        # Detection window
        window_start = (
            inc_start - max_lead_td if max_lead_td is not None else pd.Timestamp.min
        )

        for i, at in enumerate(alert_times):
            if i in matched_alerts:
                continue
            if window_start <= at < inc_start:
                detected[j] = True
                lead_s = (inc_start - at).total_seconds()
                if freq_seconds and freq_seconds > 0:
                    lead_steps = lead_s / freq_seconds
                else:
                    lead_steps = lead_s  # fallback: raw seconds
                lead_times.append(lead_steps)
                matched_alerts.add(i)
                break  # first match per incident is enough

    n_detected = sum(detected)
    fp_count = len(alert_times) - len(matched_alerts)

    # Lead time stats (in steps)
    if lead_times:
        lt_arr = np.array(lead_times)
        lt_median = float(np.median(lt_arr))
        lt_iqr = float(np.percentile(lt_arr, 75) - np.percentile(lt_arr, 25))
    else:
        lt_median = float("nan")
        lt_iqr = float("nan")

    return {
        "event_recall": n_detected / n_incidents if n_incidents else float("nan"),
        "n_detected": n_detected,
        "n_incidents": n_incidents,
        "fp_count": fp_count,
        "fp_per_10k": (fp_count / max(total_steps, 1)) * 10_000,
        "lead_times": lead_times,
        "lead_time_median_steps": lt_median,
        "lead_time_iqr_steps": lt_iqr,
    }
