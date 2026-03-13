"""PyTorch training loop with weighted BCE loss and early stopping.

The loop:
1. Computes ``pos_weight`` from class balance to handle imbalance.
2. Trains with Adam + ``BCEWithLogitsLoss``.
3. Evaluates validation PR-AUC each epoch.
4. Saves the best model state and stops early if PR-AUC does not
   improve for ``patience`` epochs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from src.training.datasets import WindowDataset
from src.utils.logging import get_logger

logger = get_logger(__name__)


def train_model(
    model: nn.Module,
    train_ds: WindowDataset,
    val_ds: WindowDataset | None,
    cfg: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]]]:
    """Train a PyTorch model and return the best checkpoint + history.

    Parameters
    ----------
    model:
        PyTorch model outputting raw logits of shape ``(batch,)``.
    train_ds, val_ds:
        Training and (optional) validation datasets.
    cfg:
        Full experiment config.

    Returns
    -------
    best_state_dict:
        ``state_dict`` of the model at the best validation epoch.
    history:
        Dict with keys ``train_loss``, ``val_loss``, ``val_pr_auc`` —
        each a list of per-epoch values.
    """
    train_cfg = cfg.get("training", {})
    epochs: int = train_cfg.get("epochs", 50)
    batch_size: int = train_cfg.get("batch_size", 64)
    lr: float = train_cfg.get("learning_rate", 1e-3)
    patience: int = train_cfg.get("early_stopping_patience", 5)

    # ── DataLoaders ──────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader: DataLoader | None = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )

    # ── Loss with class imbalance weighting ──────────────────────────
    n_pos = float(train_ds.y.sum().item())
    n_neg = float(len(train_ds) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info("pos_weight = %.2f (%.0f pos / %.0f neg).", pos_weight.item(), n_pos, n_neg)

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Training loop ────────────────────────────────────────────────
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_pr_auc": [],
    }
    best_pr_auc = -1.0
    best_state = model.state_dict()
    wait = 0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # ---- Validate ----
        val_loss = float("nan")
        val_pr_auc = float("nan")

        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_batches = 0
            all_scores: list[np.ndarray] = []
            all_labels: list[np.ndarray] = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = model(xb)
                    val_total += criterion(logits, yb).item()
                    val_batches += 1
                    all_scores.append(torch.sigmoid(logits).numpy())
                    all_labels.append(yb.numpy())

            val_loss = val_total / max(val_batches, 1)
            scores = np.concatenate(all_scores)
            labels = np.concatenate(all_labels)

            if len(np.unique(labels)) > 1:
                val_pr_auc = float(average_precision_score(labels, scores))
            else:
                val_pr_auc = 0.0

        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_pr_auc=%.4f",
            epoch,
            epochs,
            avg_train_loss,
            val_loss,
            val_pr_auc,
        )

        # ---- Early stopping ----
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    "Early stopping at epoch %d (best val_pr_auc=%.4f).",
                    epoch,
                    best_pr_auc,
                )
                break

    logger.info("Training finished. Best val_pr_auc=%.4f.", best_pr_auc)
    return best_state, history
