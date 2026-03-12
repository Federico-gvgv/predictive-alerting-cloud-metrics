"""Evaluation entry-point (stub).

Usage::

    python -m src.eval --config configs/default.yaml

Loads the config, prints the evaluation parameters, and outlines the
remaining work.  The actual model loading and metric computation are **TODO**.
"""

from __future__ import annotations

import argparse
import sys

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
    """Run the evaluation pipeline (placeholder)."""
    args = parse_args(argv)
    cfg = load_config(args.config)

    # Reproducibility
    seed: int = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)

    logger = get_logger(__name__)
    logger.info("Configuration loaded from %s", args.config)
    logger.info("Resolved config:\n%s", pretty_print_config(cfg))

    # ------------------------------------------------------------------
    # TODO – implement the following steps:
    #   1. Load test split
    #   2. Load trained model checkpoint
    #   3. Generate predictions on the test windows
    #   4. Compute regression metrics (MAE, RMSE)
    #   5. Apply alert threshold & cooldown logic
    #   6. Compute classification metrics (precision, recall, F1)
    #   7. Save evaluation report
    # ------------------------------------------------------------------
    eval_cfg = cfg.get("evaluation", {})
    cooldown = eval_cfg.get("cooldown", 10)
    threshold = eval_cfg.get("alert_threshold", 0.8)
    metrics = eval_cfg.get("metrics", [])

    logger.info(
        "Evaluation params – threshold=%.2f, cooldown=%d, metrics=%s",
        threshold,
        cooldown,
        metrics,
    )
    logger.info("Evaluation stub completed successfully – nothing to evaluate yet.")


if __name__ == "__main__":
    main()
