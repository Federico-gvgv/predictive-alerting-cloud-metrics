"""Training entry-point.

Usage::

    python -m src.train --config configs/default.yaml

Loads the dataset, builds sliding windows, splits, trains the selected
model, selects an alert threshold on the validation set, and saves
artifacts to the output directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.data.splits import time_split
from src.data.windowing import create_windows
from src.evaluation.metrics import extract_incidents_per_series
from src.evaluation.thresholding import select_threshold
from src.models import get_model
from src.utils.config import load_config, pretty_print_config
from src.utils.logging import get_logger, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a predictive-alerting model on cloud metrics.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the training pipeline."""
    args = parse_args(argv)
    cfg = load_config(args.config)

    # Reproducibility
    seed: int = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    logger = get_logger(__name__)
    logger.info("Configuration loaded from %s", args.config)
    logger.info("Resolved config:\n%s", pretty_print_config(cfg))

    # ── 1. Load dataset ──────────────────────────────────────────────
    df = load_dataset(cfg)
    logger.info("Raw dataset: %d rows.", len(df))

    # ── 2. Build sliding windows ─────────────────────────────────────
    if "series_id" in df.columns:
        freq_seconds = df.groupby("series_id")["timestamp"].diff().median().total_seconds()
    else:
        freq_seconds = df["timestamp"].diff().median().total_seconds()
    logger.info("  Sampling frequency: %.1f seconds.", freq_seconds)

    W = cfg["windowing"]["W"]
    H = cfg["windowing"]["H"]
    stride = cfg["windowing"].get("stride", 1)

    X, y, timestamps, series_ids = create_windows(df, W=W, H=H, stride=stride)

    # ── 3. Time-based splits ─────────────────────────────────────────
    split_cfg = cfg.get("split", {})
    splits = time_split(
        X,
        y,
        timestamps,
        series_ids,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
    )

    for name, (sx, sy, st, ssid) in splits.items():
        n_pos = int(sy.sum()) if len(sy) else 0
        logger.info(
            "  %-5s split: %6d windows, %5d positive (%.2f%%)",
            name,
            len(sy),
            n_pos,
            (n_pos / len(sy) * 100) if len(sy) else 0.0,
        )

    # ── 4. Train model ───────────────────────────────────────────────
    model_choice = cfg["model"]["model_choice"]
    logger.info("Training model: %s", model_choice)

    model = get_model(cfg)
    X_train, y_train, _, _ = splits["train"]
    X_val, y_val, ts_val, sid_val = splits["val"]
    model.fit(X_train, y_train, X_val, y_val)

    # ── 5. Save artifacts ────────────────────────────────────────────
    output_dir = Path(cfg.get("training", {}).get("output_dir", "outputs"))
    run_dir = output_dir / model_choice
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model.save(run_dir)

    # Save a copy of the config for reproducibility
    config_out = run_dir / "config.json"
    config_out.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    # ── 6. Threshold selection on validation set ─────────────────────
    eval_cfg = cfg.get("evaluation", {})
    cooldown = eval_cfg.get("cooldown", 10)
    target_recall = eval_cfg.get("target_event_recall", 0.8)

    logger.info("Selecting alert threshold on validation set …")
    val_scores = model.predict_proba(X_val)

    val_bounds = {}
    for sid in np.unique(sid_val):
        mask = sid_val == sid
        val_bounds[sid] = (pd.Timestamp(ts_val[mask].min()), pd.Timestamp(ts_val[mask].max()))

    val_intervals_dict = extract_incidents_per_series(df, bounds_per_series=val_bounds)
    n_val_incidents = sum(len(v) for v in val_intervals_dict.values())
    logger.info("  Validation incidents: %d total.", n_val_incidents)

    best_threshold, sweep = select_threshold(
        val_scores,
        ts_val,
        sid_val,
        val_intervals_dict,
        cooldown=cooldown,
        total_steps=len(y_val),
        target_recall=target_recall,
        max_lead_steps=H,
        freq_seconds=freq_seconds,
    )

    threshold_info = {
        "threshold": best_threshold,
        "target_event_recall": target_recall,
        "cooldown": cooldown,
        "n_val_incidents": n_val_incidents,
    }
    (run_dir / "threshold.json").write_text(
        json.dumps(threshold_info, indent=2), encoding="utf-8"
    )
    logger.info("Selected threshold: %.3f → saved to %s.", best_threshold, run_dir / "threshold.json")

    logger.info("Artifacts saved to %s.", run_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
