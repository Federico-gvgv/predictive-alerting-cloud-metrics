"""Evaluation entry-point.

Usage::

    python -m src.eval --config configs/default.yaml

Loads the dataset, builds sliding windows, splits into train / val / test,
and prints test-split statistics.  Model loading and metric computation
are **TODO**.
"""

from __future__ import annotations

import argparse

from src.data import load_dataset
from src.data.splits import time_split
from src.data.windowing import create_windows
from src.utils.config import load_config, pretty_print_config
from src.utils.logging import get_logger, set_seed


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

    # ── 4. Test-split statistics ─────────────────────────────────────
    test_X, test_y, test_ts = splits["test"]
    eval_cfg = cfg.get("evaluation", {})
    cooldown = eval_cfg.get("cooldown", 10)
    threshold = eval_cfg.get("alert_threshold", 0.8)
    metrics = eval_cfg.get("metrics", [])

    logger.info("Test split: %d windows, incident rate %.2f%%.", len(test_y), test_y.mean() * 100)
    logger.info(
        "Evaluation params – threshold=%.2f, cooldown=%d, metrics=%s",
        threshold,
        cooldown,
        metrics,
    )

    # ------------------------------------------------------------------
    # TODO – implement the following steps:
    #   5. Load trained model checkpoint
    #   6. Generate predictions on the test windows
    #   7. Compute regression metrics (MAE, RMSE)
    #   8. Apply alert threshold & cooldown logic
    #   9. Compute classification metrics (precision, recall, F1)
    #  10. Save evaluation report
    # ------------------------------------------------------------------
    logger.info("Evaluation pipeline complete – model evaluation not yet implemented.")


if __name__ == "__main__":
    main()
