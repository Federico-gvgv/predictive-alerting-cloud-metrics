"""Model registry and common interface.

Every model exposes:

* ``fit(X_train, y_train, X_val=None, y_val=None)``
* ``predict_proba(X) → np.ndarray``   – risk scores in [0, 1]
* ``save(path)`` / ``load(path)``     – artifact persistence

Use :func:`get_model` to instantiate the right class from the config.
"""

from __future__ import annotations

from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_model(cfg: dict[str, Any]):
    """Instantiate and return the model specified by ``cfg["model"]["model_choice"]``.

    Parameters
    ----------
    cfg:
        Full experiment config.

    Returns
    -------
    A model instance implementing ``fit``, ``predict_proba``, ``save``,
    ``load``.

    Raises
    ------
    ValueError
        If the requested ``model_choice`` is not recognised.
    """
    choice: str = cfg["model"]["model_choice"]

    if choice == "heuristic":
        from src.models.heuristic import HeuristicModel

        return HeuristicModel(cfg)

    if choice in ("logreg", "gbdt"):
        from src.models.logreg_baseline import SklearnBaseline

        return SklearnBaseline(cfg)

    if choice == "tcn":
        from src.models.tcn import TCNModel

        return TCNModel(cfg)

    raise ValueError(
        f"Unknown model_choice '{choice}'. "
        "Supported: heuristic, logreg, gbdt, tcn."
    )
