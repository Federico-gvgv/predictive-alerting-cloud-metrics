"""Tests for event-level metrics and cooldown logic."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import event_metrics, extract_incident_intervals
from src.evaluation.thresholding import apply_cooldown


# ── Helpers ──────────────────────────────────────────────────────────────


def _ts(minutes: list[int]) -> list[pd.Timestamp]:
    """Create timestamps at given minute offsets from a base time."""
    base = pd.Timestamp("2024-01-01")
    return [base + pd.Timedelta(minutes=m) for m in minutes]


def _intervals(pairs: list[tuple[int, int]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Create incident intervals from minute-offset pairs."""
    base = pd.Timestamp("2024-01-01")
    return [
        (base + pd.Timedelta(minutes=s), base + pd.Timedelta(minutes=e))
        for s, e in pairs
    ]


# ── Cooldown ─────────────────────────────────────────────────────────────


class TestCooldown:
    """Verify cooldown suppression logic."""

    def test_no_cooldown(self):
        """Cooldown of 0 should not suppress anything."""
        alerts = np.array([True, True, True, False, True])
        result = apply_cooldown(alerts, cooldown=0)
        np.testing.assert_array_equal(result, alerts)

    def test_basic_suppression(self):
        """Alerts within cooldown window should be suppressed."""
        alerts = np.array([True, True, True, True, True, False, False, False, True])
        result = apply_cooldown(alerts, cooldown=3)
        # t=0: fire. t=1,2,3: suppressed. t=4: fire (0+3<4). t=8: fire.
        expected = np.array([True, False, False, False, True, False, False, False, True])
        np.testing.assert_array_equal(result, expected)

    def test_no_alerts(self):
        """No alerts → no output."""
        alerts = np.array([False, False, False])
        result = apply_cooldown(alerts, cooldown=5)
        np.testing.assert_array_equal(result, [False, False, False])

    def test_single_alert(self):
        """Single alert should always pass through."""
        alerts = np.array([False, False, True, False, False])
        result = apply_cooldown(alerts, cooldown=10)
        np.testing.assert_array_equal(result, alerts)


# ── Incident interval extraction ────────────────────────────────────────


class TestIntervalExtraction:
    """Verify contiguous incident interval detection."""

    def test_single_interval(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "is_incident": [False, False, True, True, True, False, False, False, False, False],
        })
        intervals = extract_incident_intervals(df)
        assert len(intervals) == 1
        assert intervals[0][0] == df["timestamp"].iloc[2]
        assert intervals[0][1] == df["timestamp"].iloc[4]

    def test_two_intervals(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="5min"),
            "is_incident": [True, True, False, False, True, True, True, False, False, False],
        })
        intervals = extract_incident_intervals(df)
        assert len(intervals) == 2

    def test_no_incidents(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min"),
            "is_incident": [False] * 5,
        })
        assert extract_incident_intervals(df) == []

    def test_time_boundary_filtering(self):
        """Only intervals within [start_ts, end_ts] should be returned."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="5min"),
            "is_incident": [True, True, False] * 6 + [False, False],
        })
        all_intervals = extract_incident_intervals(df)
        # Filter to first 10 rows
        mid_ts = df["timestamp"].iloc[9]
        filtered = extract_incident_intervals(df, end_ts=mid_ts)
        assert len(filtered) <= len(all_intervals)


# ── Event metrics ────────────────────────────────────────────────────────


class TestEventMetrics:
    """Verify event recall, FP counting, and lead time computation."""

    def test_perfect_detection(self):
        """One alert before each incident → event recall = 1.0."""
        # Incidents at minutes 100-110 and 200-210
        intervals = _intervals([(100, 110), (200, 210)])
        # Alerts at minutes 90 and 190 (before each incident)
        alert_times = _ts([90, 190])

        result = event_metrics(alert_times, intervals, total_steps=1000)
        assert result["event_recall"] == 1.0
        assert result["n_detected"] == 2
        assert result["fp_count"] == 0

    def test_missed_incident(self):
        """Alert only before first incident → event recall = 0.5."""
        intervals = _intervals([(100, 110), (200, 210)])
        alert_times = _ts([90])  # only before first

        result = event_metrics(alert_times, intervals, total_steps=1000)
        assert result["event_recall"] == 0.5
        assert result["n_detected"] == 1

    def test_no_alerts(self):
        """No alerts at all → event recall = 0."""
        intervals = _intervals([(100, 110)])
        result = event_metrics([], intervals, total_steps=1000)
        assert result["event_recall"] == 0.0
        assert result["n_detected"] == 0

    def test_false_positives(self):
        """Alerts not matched to any incident → counted as FP."""
        intervals = _intervals([(100, 110)])
        # Alert at 90 (matched), 50 (FP — before incident but consumed by first match),
        # actually: alert at 50 fires first, matches incident. alert at 500 is FP.
        alert_times = _ts([90, 500])

        result = event_metrics(alert_times, intervals, total_steps=1000)
        assert result["n_detected"] == 1
        assert result["fp_count"] == 1
        assert result["fp_per_10k"] == (1 / 1000) * 10_000

    def test_lead_time_computation(self):
        """Lead time = incident_start - alert_time."""
        intervals = _intervals([(100, 110)])
        alert_times = _ts([80])  # 20 min before incident start

        result = event_metrics(alert_times, intervals, total_steps=1000)
        assert result["n_detected"] == 1
        # Lead time should be 20 minutes = 1200 seconds
        assert result["lead_time_median_s"] == 1200.0

    def test_no_incidents(self):
        """No incident intervals → event recall is NaN, all alerts are FP."""
        alert_times = _ts([10, 20, 30])
        result = event_metrics(alert_times, [], total_steps=1000)
        assert np.isnan(result["event_recall"])
        assert result["fp_count"] == 3

    def test_alert_during_incident_not_before(self):
        """Alert at incident start or after should NOT count as detection."""
        intervals = _intervals([(100, 110)])
        alert_times = _ts([100])  # at start, not before

        result = event_metrics(alert_times, intervals, total_steps=1000)
        assert result["n_detected"] == 0
        assert result["fp_count"] == 1
