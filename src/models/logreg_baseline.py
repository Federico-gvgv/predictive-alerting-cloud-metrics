"""Sklearn-based baseline classifiers (LogisticRegression / GradientBoosting).

The pipeline is:

1. Extract tabular features from raw windows via :mod:`src.models.features`.
2. Standardise features with ``StandardScaler``.
3. Fit an sklearn classifier with ``class_weight="balanced"`` to handle
   imbalanced classes.

Supports ``model_choice`` values: ``"logreg"`` and ``"gbdt"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.features import extract_features
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SklearnBaseline:
    """Tabular feature-based classifier.

    Parameters
    ----------
    cfg:
        Full experiment config.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.model_choice: str = cfg["model"]["model_choice"]
        model_cfg: dict[str, Any] = cfg.get("model", {})

        if self.model_choice == "logreg":
            lr_cfg = model_cfg.get("logreg", {})
            self.clf = LogisticRegression(
                C=lr_cfg.get("C", 1.0),
                max_iter=lr_cfg.get("max_iter", 1000),
                class_weight="balanced",
                solver="lbfgs",
                random_state=cfg.get("training", {}).get("seed", 42),
            )
        elif self.model_choice == "gbdt":
            gb_cfg = model_cfg.get("gbdt", {})
            self.clf = GradientBoostingClassifier(
                n_estimators=gb_cfg.get("n_estimators", 200),
                max_depth=gb_cfg.get("max_depth", 4),
                learning_rate=gb_cfg.get("learning_rate", 0.1),
                random_state=cfg.get("training", {}).get("seed", 42),
            )
        else:
            raise ValueError(f"SklearnBaseline does not support '{self.model_choice}'.")

        self.scaler = StandardScaler()
        logger.info("SklearnBaseline initialised (classifier=%s).", self.model_choice)

    # ── Interface ────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Extract features, scale, and fit the classifier.

        Parameters
        ----------
        X_train, y_train:
            Training windows and labels.
        X_val, y_val:
            Validation data (logged for info, not used for early stopping
            in the sklearn path).
        """
        logger.info("Extracting training features …")
        F_train = extract_features(X_train)
        F_train = self.scaler.fit_transform(F_train)

        logger.info("Fitting %s on %d samples …", self.model_choice, len(y_train))
        self.clf.fit(F_train, y_train)

        train_acc = self.clf.score(F_train, y_train)
        logger.info("Training accuracy: %.4f", train_acc)

        if X_val is not None and y_val is not None:
            F_val = extract_features(X_val)
            F_val = self.scaler.transform(F_val)
            val_acc = self.clf.score(F_val, y_val)
            logger.info("Validation accuracy: %.4f", val_acc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return risk scores in [0, 1].

        Parameters
        ----------
        X:
            Look-back windows, shape ``(N, W)``.

        Returns
        -------
        np.ndarray, shape ``(N,)``
        """
        F = extract_features(X)
        F = self.scaler.transform(F)

        if hasattr(self.clf, "predict_proba"):
            probs = self.clf.predict_proba(F)
            # Handle the case where only one class was seen in training
            if probs.shape[1] == 2:
                return probs[:, 1]
            return probs[:, 0]
        else:
            # GradientBoosting decision_function fallback
            return self.clf.decision_function(F)

    def save(self, path: str | Path) -> None:
        """Persist the classifier and scaler to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, path / "classifier.joblib")
        joblib.dump(self.scaler, path / "scaler.joblib")
        meta = {"model_type": "sklearn", "model_choice": self.model_choice}
        (path / "model_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("SklearnBaseline saved to %s.", path)

    @classmethod
    def load(cls, path: str | Path, cfg: dict[str, Any] | None = None) -> "SklearnBaseline":
        """Load a saved SklearnBaseline from *path*."""
        path = Path(path)
        meta = json.loads((path / "model_meta.json").read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj.model_choice = meta["model_choice"]
        obj.clf = joblib.load(path / "classifier.joblib")
        obj.scaler = joblib.load(path / "scaler.joblib")
        logger.info("SklearnBaseline loaded from %s.", path)
        return obj
