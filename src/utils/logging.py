"""Logging setup and reproducibility helpers.

* ``get_logger``  – returns a consistently-formatted stdlib logger.
* ``set_seed``    – sets seeds for ``random``, ``numpy``, and ``torch`` to
  guarantee deterministic behaviour across runs.
"""

from __future__ import annotations

import logging
import random
import sys

import numpy as np
import torch


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create (or retrieve) a logger with a clean console handler.

    Parameters
    ----------
    name:
        Logger name – typically ``__name__`` of the calling module.
    level:
        Logging level (default ``INFO``).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Seeds are set for:
    * Python's built-in ``random`` module
    * NumPy
    * PyTorch (CPU **and** CUDA, if available)

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
