# Predictive Alerting for Cloud Metrics

> **JetBrains Internship – Task #1**

Early-warning system that learns temporal patterns in cloud-infrastructure
metrics and fires alerts **before** a threshold is breached, giving operators
time to react.

---

## Quick Start

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download the NAB dataset (≈60 MB, one-time)
python scripts/download_nab.py

# 3. Train (logistic regression by default — seconds on CPU)
python -m src.train --config configs/default.yaml

# 4. Evaluate
python -m src.eval --config configs/default.yaml
```

To train the TCN instead, change `model.model_choice` in the config:

```yaml
model:
  model_choice: "tcn"   # heuristic | logreg | gbdt | tcn
```

---

## Dataset

This project uses the **[Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB)**
as its primary data source.  NAB is **not vendored** into this repository —
it is downloaded at runtime via `scripts/download_nab.py` (or automatically
on first training run if `auto_download: true` is set in the config).

For quick local tests without downloading, set `dataset.source: "synthetic"`
to generate a reproducible synthetic time series with regime shifts, spikes,
and heavy-tailed noise.

---

## Repository Structure

```
├── configs/default.yaml       # all hyper-parameters
├── scripts/download_nab.py    # dataset download helper
├── src/
│   ├── train.py               # training entry-point
│   ├── eval.py                # evaluation entry-point
│   ├── data/                  # data loading, windowing, splits
│   ├── models/                # heuristic, logreg, GBDT, TCN
│   ├── training/              # PyTorch Dataset, training loop
│   └── utils/                 # config loader, logger, seed
├── tests/
├── requirements.txt
└── README.md
```

> **Scope note.**  Only code directly relevant to the predictive-alerting
> task is included.  NAB is treated as an external dataset; no third-party
> project files are vendored.

---

## Design Decisions

### Sliding-window formulation

For each time-step *t*:

| Symbol | Definition | Meaning |
| ------ | ---------- | ------- |
| **X_t** | `value[t − W + 1 : t + 1]` | Look-back window (*W* steps) |
| **y_t** | `1` if any incident in `(t, t + H]` | Binary future-incident label |

Default: **W = 120**, **H = 30**, **stride = 1**.

### Leakage-free time-based splits

Train / val / test sets are **contiguous by time** — no shuffling.  The
look-back only uses past values; the label only scans future steps.  This
prevents any information flow from future to past across splits.

### Model progression: heuristic → logistic regression → TCN

| Model | Rationale |
| ----- | --------- |
| **Heuristic** (z-score) | Training-free floor — any learning model should beat it. |
| **Logistic regression** | Fast, interpretable linear baseline with 12 hand-crafted features; handles class imbalance via `class_weight="balanced"`. |
| **TCN** | Causal temporal convolutions with exponential dilation — captures local temporal patterns across the full look-back window.  CPU-friendly (~36 k params default).  No manual feature engineering. |

### Alerting metrics beyond ROC-AUC

ROC-AUC can be misleading on imbalanced data.  The evaluation pipeline also
reports **PR-AUC** (precision–recall area), and threshold-based **precision /
recall / F1** — metrics that directly reflect alerting quality in a scenario
where false alarms are costly but missed incidents are worse.

---

## Configuration

All hyper-parameters live in [`configs/default.yaml`](configs/default.yaml):

| Section      | Controls                                           |
| ------------ | -------------------------------------------------- |
| `dataset`    | NAB / synthetic source, subset, auto-download      |
| `windowing`  | W, H, stride                                       |
| `split`      | Train / val / test ratios                          |
| `training`   | Epochs, batch size, LR, patience, output dir       |
| `evaluation` | Alert threshold, cooldown, metric list             |
| `model`      | Model choice + per-model hyper-parameters          |

---

## Running Tests

```bash
pytest tests/ -v
```

## License

This project is part of a JetBrains internship application task.