"""Heuristic baseline – z-score spike detector.

For each look-back window of length *W* the model computes the z-score of the
**last** value relative to the window mean / std, then maps it to a risk score
via ``sigmoid(|z| − z_threshold)``.

This is a **training-free** baseline: ``fit()`` is a no-op.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class HeuristicModel:
    """Z-score spike detector (no learnable parameters).

    Parameters
    ----------
    cfg:
        Full experiment config.  Reads ``cfg["model"]["heuristic"]["z_threshold"]``
        (default 2.0).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        heur_cfg = cfg.get("model", {}).get("heuristic", {})
        self.z_threshold: float = heur_cfg.get("z_threshold", 2.0)
        logger.info(
            "HeuristicModel initialised (z_threshold=%.2f).", self.z_threshold
        )

    # ── Interface ────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """No-op – the heuristic has no learnable parameters."""
        logger.info("HeuristicModel.fit() – nothing to train.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return risk scores in [0, 1] for each window.

        Parameters
        ----------
        X:
            Look-back windows, shape ``(N, W)``.

        Returns
        -------
        np.ndarray, shape ``(N,)``
        """
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        std = np.where(std < 1e-8, 1e-8, std)  # avoid division by zero
        last = X[:, -1]

        z = np.abs((last - mean) / std)
        scores = _sigmoid(z - self.z_threshold)
        return scores

    def save(self, path: str | Path) -> None:
        """Persist model metadata to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {"model_type": "heuristic", "z_threshold": self.z_threshold}
        (path / "model_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("HeuristicModel saved to %s.", path)

    @classmethod
    def load(cls, path: str | Path, cfg: dict[str, Any] | None = None) -> "HeuristicModel":
        """Load a saved HeuristicModel from *path*."""
        path = Path(path)
        meta = json.loads((path / "model_meta.json").read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj.z_threshold = meta["z_threshold"]
        logger.info("HeuristicModel loaded from %s.", path)
        return obj
