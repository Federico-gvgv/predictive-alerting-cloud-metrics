"""Lightweight Temporal Convolutional Network (TCN) for binary classification.

Architecture
------------
Input ``(batch, 1, W)`` → stack of :class:`TemporalBlock` layers with
exponentially increasing dilation → global average pooling → linear →
sigmoid → scalar probability.

Causal convolutions guarantee the model only sees past values within each
window – no future leakage.

References
----------
Bai, Kolter & Koltun (2018), *An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Building blocks ─────────────────────────────────────────────────────


class _Chomp1d(nn.Module):
    """Remove trailing padding so that the convolution is strictly causal."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """One residual block: two causal-conv layers + optional 1×1 skip.

    Parameters
    ----------
    in_ch:  input channels
    out_ch: output channels
    kernel_size: convolution kernel width
    dilation: dilation factor
    dropout: dropout probability
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 1×1 conv to match dimensions for the residual if needed
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.skip(x))


class TCNClassifier(nn.Module):
    """Stack of :class:`TemporalBlock` layers → global-avg-pool → linear.

    Parameters
    ----------
    in_channels:  number of input channels (1 for univariate).
    channels:     number of hidden channels per block.
    kernel_size:  temporal kernel width.
    num_levels:   number of TemporalBlock layers (dilation = 2^i).
    dropout:      dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 32,
        kernel_size: int = 7,
        num_levels: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else channels
            layers.append(
                TemporalBlock(in_ch, channels, kernel_size, dilation=2**i, dropout=dropout)
            )
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:  ``(batch, 1, W)``

        Returns
        -------
        logits: ``(batch,)``
        """
        h = self.tcn(x)                   # (batch, channels, W)
        h = h.mean(dim=2)                 # global average pooling → (batch, channels)
        return self.head(h).squeeze(-1)    # (batch,)


# ── Wrapper implementing the project model interface ────────────────────


class TCNModel:
    """High-level wrapper around :class:`TCNClassifier`.

    Implements ``fit``, ``predict_proba``, ``save``, ``load`` so that it
    integrates transparently with ``train.py`` and ``eval.py``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        tcn_cfg = cfg.get("model", {}).get("tcn", {})
        self.channels: int = tcn_cfg.get("channels", 32)
        self.kernel_size: int = tcn_cfg.get("kernel_size", 7)
        self.num_levels: int = tcn_cfg.get("num_levels", 3)
        self.dropout: float = tcn_cfg.get("dropout", 0.1)
        self.cfg = cfg

        self.net = TCNClassifier(
            in_channels=1,
            channels=self.channels,
            kernel_size=self.kernel_size,
            num_levels=self.num_levels,
            dropout=self.dropout,
        )

        n_params = sum(p.numel() for p in self.net.parameters())
        logger.info(
            "TCNModel initialised – %d levels, %d channels, kernel=%d (%d params).",
            self.num_levels,
            self.channels,
            self.kernel_size,
            n_params,
        )

    # ── Interface ────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Train the TCN using the PyTorch training loop."""
        from src.training.datasets import WindowDataset
        from src.training.loops import train_model

        train_ds = WindowDataset(X_train, y_train)
        val_ds = WindowDataset(X_val, y_val) if X_val is not None else None

        best_state, history = train_model(self.net, train_ds, val_ds, self.cfg)
        self.net.load_state_dict(best_state)
        self.history = history

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return risk scores in [0, 1]."""
        self.net.eval()
        t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N, 1, W)
        logits = self.net(t)
        return torch.sigmoid(logits).numpy()

    def save(self, path: str | Path) -> None:
        """Save checkpoint and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "tcn_checkpoint.pt")
        meta = {
            "model_type": "tcn",
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "num_levels": self.num_levels,
            "dropout": self.dropout,
        }
        (path / "model_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("TCNModel saved to %s.", path)

    @classmethod
    def load(cls, path: str | Path, cfg: dict[str, Any] | None = None) -> "TCNModel":
        """Load a saved TCNModel."""
        path = Path(path)
        meta = json.loads((path / "model_meta.json").read_text(encoding="utf-8"))

        # Build a minimal cfg from saved metadata so we can reconstruct
        if cfg is None:
            cfg = {"model": {"tcn": meta}}
        else:
            cfg = dict(cfg)
            cfg.setdefault("model", {})["tcn"] = {
                "channels": meta["channels"],
                "kernel_size": meta["kernel_size"],
                "num_levels": meta["num_levels"],
                "dropout": meta["dropout"],
            }

        obj = cls(cfg)
        state = torch.load(path / "tcn_checkpoint.pt", map_location="cpu", weights_only=True)
        obj.net.load_state_dict(state)
        logger.info("TCNModel loaded from %s.", path)
        return obj
