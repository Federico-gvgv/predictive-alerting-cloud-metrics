"""Evaluation plots: PR curve, lead-time histogram, threshold sweep.

All plots are saved as PNG files.  Matplotlib is used with a clean,
publication-ready style.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    save_path: str | Path,
) -> None:
    """Plot and save a precision–recall curve."""
    save_path = _ensure_dir(Path(save_path))

    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class – skipping PR curve.")
        return

    precision, recall, _ = precision_recall_curve(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, color="#2563eb")
    ax.fill_between(recall, precision, alpha=0.15, color="#2563eb")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("PR curve saved to %s.", save_path)


def plot_lead_time_histogram(
    lead_times: list[float],
    save_path: str | Path,
) -> None:
    """Plot and save a histogram of lead times (in seconds)."""
    save_path = _ensure_dir(Path(save_path))

    if not lead_times:
        logger.warning("No lead times – skipping histogram.")
        return

    lt = np.array(lead_times)
    # Convert to minutes for readability
    lt_min = lt / 60.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lt_min, bins=min(20, max(5, len(lt_min) // 3)), color="#10b981",
            edgecolor="white", linewidth=0.8)
    ax.axvline(np.median(lt_min), color="#dc2626", linestyle="--", linewidth=2,
               label=f"Median: {np.median(lt_min):.1f} min")
    ax.set_xlabel("Lead Time (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Lead Time Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Lead time histogram saved to %s.", save_path)


def plot_threshold_sweep(
    sweep: list[dict],
    save_path: str | Path,
) -> None:
    """Plot event recall and FP/10k vs threshold."""
    save_path = _ensure_dir(Path(save_path))

    if not sweep:
        logger.warning("Empty sweep – skipping plot.")
        return

    thresholds = [s["threshold"] for s in sweep]
    recalls = [r if r == r else 0.0 for r in (s["event_recall"] for s in sweep)]
    fps = [s["fp_per_10k"] for s in sweep]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color_recall = "#2563eb"
    color_fp = "#dc2626"

    ax1.plot(thresholds, recalls, color=color_recall, linewidth=2, label="Event Recall")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Event Recall", color=color_recall)
    ax1.tick_params(axis="y", labelcolor=color_recall)
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, fps, color=color_fp, linewidth=2, linestyle="--",
             label="FP / 10k steps")
    ax2.set_ylabel("FP per 10k Steps", color=color_fp)
    ax2.tick_params(axis="y", labelcolor=color_fp)

    ax1.set_title("Threshold Sweep: Event Recall vs False Positives")
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Threshold sweep plot saved to %s.", save_path)
