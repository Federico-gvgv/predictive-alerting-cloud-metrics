"""Data-loading dispatcher.

Call :func:`load_dataset` with the full experiment config to obtain a
``pd.DataFrame`` with columns ``["timestamp", "value", "is_incident"]``.
The ``source`` key in ``cfg["dataset"]`` selects NAB or synthetic data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load the dataset specified by *cfg* and return a unified DataFrame.

    Parameters
    ----------
    cfg:
        Full experiment config (parsed YAML).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp`` (datetime64), ``value`` (float),
        ``is_incident`` (bool).

    Raises
    ------
    ValueError
        If ``cfg["dataset"]["source"]`` is not ``"nab"`` or ``"synthetic"``.
    """
    source: str = cfg["dataset"].get("source", "nab")

    if source == "nab":
        from src.data.nab import load_nab

        logger.info("Loading NAB dataset …")
        try:
            df = load_nab(cfg)
        except Exception as exc:
            logger.warning("NAB loading failed (%s). Falling back to synthetic.", exc)
            from src.data.synthetic import generate_synthetic

            df = generate_synthetic(cfg)
    elif source == "synthetic":
        from src.data.synthetic import generate_synthetic

        logger.info("Generating synthetic dataset …")
        df = generate_synthetic(cfg)
    else:
        raise ValueError(
            f"Unknown dataset source '{source}'. Choose 'nab' or 'synthetic'."
        )

    logger.info(
        "Dataset ready – %d rows, incident rate %.2f%%.",
        len(df),
        df["is_incident"].mean() * 100,
    )
    return df
