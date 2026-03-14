"""Tests for the sliding-window labeling logic in src.data.windowing."""

import numpy as np
import pandas as pd
import pytest

from src.data.windowing import create_windows


def _make_df(values: list[float], incidents: list[bool]) -> pd.DataFrame:
    """Build a minimal DataFrame for windowing tests."""
    n = len(values)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "value": values,
        "is_incident": incidents,
    })


# ── Basic label correctness ─────────────────────────────────────────────


class TestFutureHorizonLabeling:
    """Verify y_t = 1 iff any incident in (t, t+H]."""

    def test_no_incidents(self):
        """All labels should be 0 when there are no incidents."""
        df = _make_df(
            values=[1.0] * 20,
            incidents=[False] * 20,
        )
        _, y, _, _ = create_windows(df, W=5, H=3, stride=1)
        assert y.sum() == 0, "Expected all labels to be 0."

    def test_all_incidents(self):
        """All labels should be 1 when every step is an incident."""
        df = _make_df(
            values=[1.0] * 20,
            incidents=[True] * 20,
        )
        _, y, _, _ = create_windows(df, W=5, H=3, stride=1)
        assert y.sum() == len(y), "Expected all labels to be 1."

    def test_single_future_incident(self):
        """A single incident at the right position should label correctly."""
        # 10 steps, W=3, H=3 → anchors at t=2..6
        # Place incident at step 5 → labels at t where 5 in (t, t+3]
        # i.e. t=2 (looks at 3,4,5 ✓), t=3 (looks at 4,5,6 ✓),
        #      t=4 (looks at 5,6,7 ✓), t=5 (looks at 6,7,8 ✗),
        #      t=6 (looks at 7,8,9 ✗)
        incidents = [False] * 10
        incidents[5] = True
        df = _make_df([1.0] * 10, incidents)

        _, y, _, _ = create_windows(df, W=3, H=3, stride=1)

        # anchors: t=2,3,4,5,6 → labels: 1,1,1,0,0
        expected = [1, 1, 1, 0, 0]
        np.testing.assert_array_equal(y, expected)

    def test_incident_exactly_at_boundary(self):
        """Incident at t+H should be included (label scans (t, t+H])."""
        incidents = [False] * 10
        incidents[6] = True  # step 6
        df = _make_df([1.0] * 10, incidents)

        # W=3, H=3, anchors t=2..6
        # t=3: horizon (3,6] includes step 6 → label 1
        _, y, _, _ = create_windows(df, W=3, H=3, stride=1)

        # anchors: t=2(3,4,5→0), t=3(4,5,6→1), t=4(5,6,7→1), t=5(6,7,8→1), t=6(7,8,9→0)
        expected = [0, 1, 1, 1, 0]
        np.testing.assert_array_equal(y, expected)

    def test_incident_at_t_not_in_future(self):
        """Incident exactly at t should NOT cause y_t=1 (we scan (t, t+H])."""
        incidents = [False] * 10
        incidents[4] = True
        df = _make_df([1.0] * 10, incidents)

        _, y, _, _ = create_windows(df, W=3, H=3, stride=1)
        # anchor t=4 scans (4,7] = steps 5,6,7 → no incident → y=0
        assert y[2] == 0, "Incident at t should not cause y_t=1."

    def test_lookback_uses_correct_values(self):
        """Each X row should contain exactly the W values ending at t."""
        values = list(range(10))
        df = _make_df(values=[float(v) for v in values], incidents=[False] * 10)

        X, _, _, _ = create_windows(df, W=4, H=2, stride=1)
        # First anchor t=3 → X[0] = [0,1,2,3]
        np.testing.assert_array_equal(X[0], [0.0, 1.0, 2.0, 3.0])
        # Second anchor t=4 → X[1] = [1,2,3,4]
        np.testing.assert_array_equal(X[1], [1.0, 2.0, 3.0, 4.0])

    def test_stride(self):
        """Stride > 1 should skip windows."""
        df = _make_df([1.0] * 20, [False] * 20)
        X1, _, _, _ = create_windows(df, W=5, H=3, stride=1)
        X2, _, _, _ = create_windows(df, W=5, H=3, stride=3)
        assert len(X2) < len(X1)
        assert len(X2) == len(range(4, 17, 3))  # anchors 4..16 step 3

    def test_timestamps_align(self):
        """Returned timestamps should match the anchor positions."""
        df = _make_df([1.0] * 10, [False] * 10)
        _, _, ts, _ = create_windows(df, W=3, H=3, stride=1)
        # First anchor t=2, last t=6
        assert ts[0] == df["timestamp"].iloc[2]
        assert ts[-1] == df["timestamp"].iloc[6]

    def test_too_short_raises(self):
        """Series shorter than W+H should raise ValueError."""
        df = _make_df([1.0] * 5, [False] * 5)
        with pytest.raises(ValueError, match="No series were long enough to form windows."):
            create_windows(df, W=3, H=5, stride=1)

    def test_multi_series_windowing(self):
        """Windows from different series should never mix."""
        df1 = _make_df([1.0] * 10, [False] * 10)
        df1["series_id"] = "A"
        df2 = _make_df([2.0] * 10, [False] * 10)
        df2["series_id"] = "B"
        df_combined = pd.concat([df1, df2])

        # W=3, H=3 -> each series produces 5 windows
        X, y, ts, sid = create_windows(df_combined, W=3, H=3, stride=1)

        assert len(X) == 10
        # First 5 windows from A should have value 1.0
        assert (X[:5] == 1.0).all()
        assert (sid[:5] == "A").all()
        # Next 5 windows from B should have value 2.0
        assert (X[5:] == 2.0).all()
        assert (sid[5:] == "B").all()
