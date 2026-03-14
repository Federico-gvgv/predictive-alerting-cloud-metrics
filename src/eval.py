"""Evaluation entry-point.

Usage::

    python -m src.eval --config configs/default.yaml

Loads the dataset, builds windows, loads the trained model and selected
threshold, computes pointwise + event-level alerting metrics, generates
plots, and saves a comprehensive JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.data.splits import time_split
from src.data.windowing import create_windows
from src.evaluation.metrics import (
    event_metrics,
    extract_incident_intervals,
    pointwise_metrics,
)
from src.evaluation.plots import (
    plot_lead_time_histogram,
    plot_pr_curve,
    plot_threshold_sweep,
)
from src.evaluation.thresholding import apply_cooldown, select_threshold
from src.utils.config import load_config, pretty_print_config
from src.utils.logging import get_logger, set_seed


def _load_model(run_dir: Path, cfg: dict):
    """Load a model from *run_dir* by reading its ``model_meta.json``."""
    meta_path = run_dir / "model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"No model_meta.json found in {run_dir}. Run training first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_type = meta.get("model_type", "")

    if model_type == "heuristic":
        from src.models.heuristic import HeuristicModel
        return HeuristicModel.load(run_dir, cfg)

    if model_type == "sklearn":
        from src.models.logreg_baseline import SklearnBaseline
        return SklearnBaseline.load(run_dir, cfg)

    if model_type == "tcn":
        from src.models.tcn import TCNModel
        return TCNModel.load(run_dir, cfg)

    raise ValueError(f"Unknown model_type '{model_type}' in {meta_path}.")


def _compute_event_results(
    scores: np.ndarray,
    timestamps: pd.DatetimeIndex,
    incident_intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    threshold: float,
    cooldown: int,
    total_steps: int,
    label: str,
    logger,
    max_lead_steps: int | None = None,
    freq_seconds: float | None = None,
) -> dict:
    """Compute event metrics with a given cooldown and log them."""
    raw_alerts = scores >= threshold
    if cooldown > 0:
        alerts = apply_cooldown(raw_alerts, cooldown)
    else:
        alerts = raw_alerts

    alert_times = [timestamps[i] for i in range(len(alerts)) if alerts[i]]
    result = event_metrics(
        alert_times, incident_intervals, total_steps,
        max_lead_steps=max_lead_steps, freq_seconds=freq_seconds,
    )

    logger.info("── Event Metrics (%s) ──", label)
    logger.info("  %-28s %.4f", "event_recall", result["event_recall"])
    logger.info("  %-28s %d / %d", "detected / incidents",
                result["n_detected"], result["n_incidents"])
    logger.info("  %-28s %d", "fp_count", result["fp_count"])
    logger.info("  %-28s %.2f", "fp_per_10k", result["fp_per_10k"])
    logger.info("  %-28s %.1f steps", "lead_time_median",
                result["lead_time_median_steps"])
    logger.info("  %-28s %.1f steps", "lead_time_iqr",
                result["lead_time_iqr_steps"])

    # Strip raw lead_times list for JSON (keep summary stats)
    return {k: v for k, v in result.items() if k != "lead_times"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a predictive-alerting model on cloud metrics.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the evaluation pipeline."""
    args = parse_args(argv)
    cfg = load_config(args.config)

    # Reproducibility
    seed: int = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    logger = get_logger(__name__)
    logger.info("Configuration loaded from %s", args.config)
    logger.info("Resolved config:\n%s", pretty_print_config(cfg))

    # ── 1. Load dataset & windows ────────────────────────────────────
    df = load_dataset(cfg)
    logger.info("Raw dataset: %d rows.", len(df))
    freq_seconds = df["timestamp"].diff().median().total_seconds()

    W = cfg["windowing"]["W"]
    H = cfg["windowing"]["H"]
    stride = cfg["windowing"].get("stride", 1)

    X, y, timestamps = create_windows(df, W=W, H=H, stride=stride)

    split_cfg = cfg.get("split", {})
    splits = time_split(
        X,
        y,
        timestamps,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
    )

    test_X, test_y, test_ts = splits["test"]
    logger.info("Test split: %d windows, incident rate %.2f%%.",
                len(test_y), test_y.mean() * 100)

    # ── 2. Load trained model ────────────────────────────────────────
    model_choice = cfg["model"]["model_choice"]
    output_dir = Path(cfg.get("training", {}).get("output_dir", "outputs"))
    run_dir = output_dir / model_choice

    logger.info("Loading model from %s …", run_dir)
    model = _load_model(run_dir, cfg)

    # ── 3. Load threshold ────────────────────────────────────────────
    eval_cfg = cfg.get("evaluation", {})
    cooldown = eval_cfg.get("cooldown", 10)

    threshold_path = run_dir / "threshold.json"
    if threshold_path.is_file():
        thr_info = json.loads(threshold_path.read_text(encoding="utf-8"))
        threshold = thr_info["threshold"]
        logger.info("Loaded selected threshold: %.3f (from %s).", threshold, threshold_path)
    else:
        threshold = eval_cfg.get("alert_threshold", 0.5)
        logger.info("No threshold.json found – using config default: %.3f.", threshold)

    # ── 4. Predict ───────────────────────────────────────────────────
    scores = model.predict_proba(test_X)

    # ── 5. Pointwise metrics ─────────────────────────────────────────
    pw = pointwise_metrics(test_y, scores, threshold)

    logger.info("── Pointwise Metrics ──")
    for k, v in pw.items():
        logger.info("  %-24s %.4f", k, v)

    # ── 6. Event-level metrics ───────────────────────────────────────
    test_intervals = extract_incident_intervals(
        df,
        start_ts=pd.Timestamp(test_ts.min()),
        end_ts=pd.Timestamp(test_ts.max()),
    )
    logger.info("Test incident intervals: %d.", len(test_intervals))

    # With cooldown
    ev_cd = _compute_event_results(
        scores, test_ts, test_intervals, threshold, cooldown,
        len(test_y), f"cooldown={cooldown}", logger,
        max_lead_steps=H, freq_seconds=freq_seconds,
    )
    # Without cooldown
    ev_nocd = _compute_event_results(
        scores, test_ts, test_intervals, threshold, 0,
        len(test_y), "no cooldown", logger,
        max_lead_steps=H, freq_seconds=freq_seconds,
    )

    # ── 7. Threshold sweep on test (for plotting only) ───────────────
    _, test_sweep = select_threshold(
        scores, test_ts, test_intervals,
        cooldown=cooldown, total_steps=len(test_y),
        target_recall=eval_cfg.get("target_event_recall", 0.8),
        max_lead_steps=H, freq_seconds=freq_seconds,
    )

    # ── 8. Plots ─────────────────────────────────────────────────────
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_pr_curve(test_y, scores, plot_dir / "pr_curve.png")

    # Lead times (recompute with raw lead times)
    raw_alerts = apply_cooldown(scores >= threshold, cooldown)
    alert_times = [test_ts[i] for i in range(len(raw_alerts)) if raw_alerts[i]]
    ev_full = event_metrics(
        alert_times, test_intervals, len(test_y),
        max_lead_steps=H, freq_seconds=freq_seconds,
    )
    plot_lead_time_histogram(ev_full["lead_times"], plot_dir / "lead_time.png")

    plot_threshold_sweep(test_sweep, plot_dir / "threshold_sweep.png")

    # ── 9. Save report ───────────────────────────────────────────────
    report = {
        "model": model_choice,
        "threshold": threshold,
        "cooldown": cooldown,
        "n_test": int(len(test_y)),
        "n_positive": int(test_y.sum()),
        "n_test_incidents": len(test_intervals),
        "pointwise": pw,
        "event_with_cooldown": ev_cd,
        "event_without_cooldown": ev_nocd,
    }

    report_path = run_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Full evaluation report saved to %s.", report_path)
    logger.info("Plots saved to %s.", plot_dir)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
