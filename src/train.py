"""Training entry-point.

Usage::

    python -m src.train --config configs/default.yaml

Loads the dataset, builds sliding windows, splits, trains the selected
model, and saves artifacts to the output directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.data import load_dataset
from src.data.splits import time_split
from src.data.windowing import create_windows
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
    W = cfg["windowing"]["W"]
    H = cfg["windowing"]["H"]
    stride = cfg["windowing"].get("stride", 1)

    X, y, timestamps = create_windows(df, W=W, H=H, stride=stride)

    # ── 3. Time-based splits ─────────────────────────────────────────
    split_cfg = cfg.get("split", {})
    splits = time_split(
        X,
        y,
        timestamps,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
    )

    for name, (sx, sy, st) in splits.items():
        logger.info(
            "  %s: %d windows, incident rate %.2f%%",
            name,
            len(sy),
            sy.mean() * 100,
        )

    # ── 4. Train model ───────────────────────────────────────────────
    model_choice = cfg["model"]["model_choice"]
    logger.info("Training model: %s", model_choice)

    model = get_model(cfg)
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
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

    logger.info("Artifacts saved to %s.", run_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
