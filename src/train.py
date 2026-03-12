"""Training entry-point (stub).

Usage::

    python -m src.train --config configs/default.yaml

This module wires together config loading, seed setting, and logging.
The actual data pipeline and model training loop are **TODO**.
"""

from __future__ import annotations

import argparse
import sys

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
    """Run the training pipeline (placeholder)."""
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
    #   1. Load & preprocess the dataset  (src.data)
    #   2. Build sliding windows          (W=%d, H=%d)
    #   3. Create train / val / test splits
    #   4. Instantiate the model           (model_choice=%s)
    #   5. Training loop with early stopping
    #   6. Save best checkpoint to output_dir
    # ------------------------------------------------------------------
    w = cfg["windowing"]["W"]
    h = cfg["windowing"]["H"]
    model_choice = cfg["model"]["model_choice"]

    logger.info(
        "Next steps: implement data pipeline (W=%d, H=%d) and %s model training.",
        w,
        h,
        model_choice,
    )
    logger.info("Training stub completed successfully – nothing to train yet.")


if __name__ == "__main__":
    main()
