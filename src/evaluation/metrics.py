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


def extract_incidents_per_series(
    df: pd.DataFrame,
    bounds_per_series: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Extract contiguous incident intervals per series from the raw dataframe.

    Parameters
    ----------
    df:
        DataFrame with ``timestamp``, ``is_incident``, and ``series_id`` columns.
    bounds_per_series:
        Optional time boundaries ``(start, end)`` per ``series_id`` to restrict extraction.

    Returns
    -------
    dict mapping ``series_id`` to a list of ``(interval_start, interval_end)`` tuples.
    """
    res = {}
    if "series_id" not in df.columns:
        df = df.copy()
        df["series_id"] = "default"
        
    for sid, group in df.groupby("series_id", sort=False):
        sub = group.copy()
        if bounds_per_series and sid in bounds_per_series:
            start_ts, end_ts = bounds_per_series[sid]
            sub = sub[(sub["timestamp"] >= start_ts) & (sub["timestamp"] <= end_ts)]
            
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
                
        if in_incident:
            intervals.append((pd.Timestamp(ts[i_start]), pd.Timestamp(ts[len(incident) - 1])))
            
        res[sid] = intervals
        
    return res


# ── Event-level metrics ─────────────────────────────────────────────────


def event_metrics(
    alert_times_dict: dict[str, list[pd.Timestamp]],
    incidents_dict: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
    total_steps: int,
    max_lead_steps: int | None = None,
    freq_seconds: float | None = None,
) -> dict[str, object]:
    """Compute event-level alerting metrics aggregated across multiple series.

    An incident is **detected** if at least one alert fires strictly
    *before* the incident's ``start_ts`` **and** within the maximum lead
    window for the same ``series_id``.
    
    Parameters
    ----------
    alert_times_dict:
        Dict mapping series_id to sorted lists of alert timestamps.
    incidents_dict:
        Dict mapping series_id to list of ``(start, end)`` intervals.
    total_steps:
        Total number of time-steps combined across all series.
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
    n_incidents = sum(len(ivs) for ivs in incidents_dict.values())
    if n_incidents == 0:
        fp_count = sum(len(at) for at in alert_times_dict.values())
        return {
            "event_recall": float("nan"),
            "n_detected": 0,
            "n_incidents": 0,
            "fp_count": fp_count,
            "fp_per_10k": (fp_count / max(total_steps, 1)) * 10_000,
            "lead_times": [],
            "lead_time_median_steps": float("nan"),
            "lead_time_iqr_steps": float("nan"),
        }

    if max_lead_steps is not None and freq_seconds is not None:
        max_lead_td = pd.Timedelta(seconds=max_lead_steps * freq_seconds)
    else:
        max_lead_td = None

    n_detected = 0
    fp_count = 0
    lead_times: list[float] = []

    for sid, intervals in incidents_dict.items():
        alert_times = alert_times_dict.get(sid, [])
        detected = [False] * len(intervals)
        matched_alerts: set[int] = set()

        for j, (inc_start, inc_end) in enumerate(intervals):
            window_start = (
                inc_start - max_lead_td if max_lead_td is not None else pd.Timestamp.min
            )

            for i, at in enumerate(alert_times):
                if i in matched_alerts:
                    continue
                if window_start <= at < inc_start:
                    detected[j] = True
                    lead_s = (inc_start - at).total_seconds()
                    lead_steps = lead_s / freq_seconds if (freq_seconds and freq_seconds > 0) else lead_s
                    lead_times.append(lead_steps)
                    matched_alerts.add(i)
                    break

        n_detected += sum(detected)
        fp_count += len(alert_times) - len(matched_alerts)

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
