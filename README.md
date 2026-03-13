# Predictive Alerting for Cloud Metrics

> **JetBrains Internship – Task #1**

Early-warning system that forecasts cloud-infrastructure metrics (CPU, memory,
network I/O, …) and fires alerts **before** a threshold is breached, giving
operators time to react.

---

## Repository Structure

```
predictive-alerting-cloud-metrics/
├── configs/
│   └── default.yaml          # experiment configuration
├── src/
│   ├── __init__.py
│   ├── train.py               # training entry-point
│   ├── eval.py                # evaluation entry-point
│   ├── data/
│   │   ├── __init__.py        # load_dataset() dispatcher
│   │   ├── nab.py             # NAB downloader & loader
│   │   ├── synthetic.py       # synthetic fallback generator
│   │   ├── windowing.py       # sliding-window extraction
│   │   └── splits.py          # time-based train/val/test splits
│   ├── models/
│   │   ├── __init__.py        # get_model() factory
│   │   ├── heuristic.py       # z-score spike detector
│   │   ├── features.py        # tabular feature extractor
│   │   └── logreg_baseline.py # LogisticRegression / GBDT
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # YAML config loader
│       └── logging.py         # logger & seed helpers
├── tests/
│   └── __init__.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train a model

```bash
python -m src.train --config configs/default.yaml
```

### 4. Evaluate the model

```bash
python -m src.eval --config configs/default.yaml
```

---

## Dataset Setup

### NAB (default)

The [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) is used as the
primary dataset.  On first run the pipeline **auto-downloads** the NAB
repository as a zip archive and extracts it into `data/nab/`.

| Config key                 | Default            | Description                                  |
| -------------------------- | ------------------ | -------------------------------------------- |
| `dataset.source`           | `"nab"`            | Select NAB or synthetic                      |
| `dataset.nab.data_dir`     | `"data/nab"`       | Local directory for the NAB tree             |
| `dataset.nab.subset`       | `"realKnownCause"` | Subdirectory under `NAB/data/`               |
| `dataset.nab.files`        | `null`             | Explicit file list, or `null` for all        |
| `dataset.nab.auto_download`| `true`             | Download automatically if data is missing    |

The default subset (`realKnownCause`) contains 7 CSV files (~15 k rows) so
that a full CPU run finishes well within 10 minutes.

### Synthetic fallback

Set `dataset.source: "synthetic"` for a quick local test without downloading.

---

## Windowing & Labelling

| Symbol | Definition | Description |
| ------ | ---------- | ----------- |
| **X_t** | `value[t − W + 1 : t + 1]` | Look-back window of *W* time-steps |
| **y_t** | `1 if any incident in (t, t + H]` | Binary future-incident label |

### Leakage avoidance

The train / val / test split is **strictly contiguous by time** – windows are
never shuffled.  The look-back only uses past values; the label only scans
future steps.

---

## Baseline Models

The project ships three baseline models.  Select one via
`model.model_choice` in the config:

### `heuristic` – Z-score spike detector

A training-free baseline.  For each window the model computes the z-score of
the last value relative to the window mean/std and maps it to a risk score
via `sigmoid(|z| − z_threshold)`.

*Why?*  Establishes a floor – any learning-based model should beat this
simple statistical rule.

### `logreg` – Logistic Regression *(default)*

Extracts 12 tabular features per window (mean, std, min, max, slope,
quantiles, EWMA ratio, last value, delta from mean), standardises them,
and trains a `LogisticRegression` with `class_weight="balanced"`.

*Why?*  A strong, interpretable linear baseline that handles class imbalance
and runs in seconds.

### `gbdt` – Gradient Boosting

Same feature pipeline as `logreg` but trains a `GradientBoostingClassifier`.
Generally stronger than logistic regression at the cost of longer training.

### Selecting a model

```yaml
# configs/default.yaml
model:
  model_choice: "logreg"   # or "heuristic" or "gbdt"
```

### Evaluation metrics

After training, `python -m src.eval` computes:

- **ROC-AUC** – ranking quality across all thresholds
- **PR-AUC** – precision–recall trade-off (better for imbalanced data)
- **Precision / Recall / F1** at the configured `alert_threshold`

Results are saved to `outputs/<model_choice>/eval_report.json`.

---

## Configuration

All hyper-parameters live in `configs/default.yaml`. Key sections:

| Section      | Purpose                                            |
| ------------ | -------------------------------------------------- |
| `dataset`    | Data source, NAB / synthetic options               |
| `windowing`  | Sliding-window sizes (`W`, `H`) and stride         |
| `split`      | Train / val / test ratios                          |
| `training`   | Epochs, batch size, learning rate, early stopping  |
| `evaluation` | Alert threshold, cooldown, evaluation metrics      |
| `model`      | Model choice and per-model hyper-parameters        |

## Running Tests

```bash
pytest tests/ -v
```

## License

This project is part of a JetBrains internship application task.