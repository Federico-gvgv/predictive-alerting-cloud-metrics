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

### 3. Run the training pipeline

```bash
python -m src.train --config configs/default.yaml
```

### 4. Run the evaluation pipeline

```bash
python -m src.eval --config configs/default.yaml
```

Both commands load the dataset (NAB by default, with auto-download), build
sliding windows, split the data, and print dataset statistics.  Model training
and evaluation are still placeholder TODOs.

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

If NAB is unavailable or you want a quick local test, set
`dataset.source: "synthetic"`:

```yaml
dataset:
  source: "synthetic"
  synthetic:
    n_samples: 5000
    freq: "5min"
    n_regimes: 3
    n_spikes: 10
```

The generator produces a time series with a smooth trend, daily seasonality,
heavy-tailed noise (Student-*t*), regime shifts, and spike injections.  Regime
shifts and spikes are labelled as **incidents**.

---

## Windowing & Labelling

For each anchor time-step *t* the pipeline constructs:

| Symbol | Definition | Description |
| ------ | ---------- | ----------- |
| **X_t** | `value[t − W + 1 : t + 1]` | Look-back window of *W* time-steps |
| **y_t** | `1 if any incident in (t, t + H]` | Binary future-incident label |

* **W** (look-back) and **H** (horizon) are set in `configs/default.yaml`
  under `windowing` (defaults: W = 120, H = 30).
* An optional **stride** controls the step between consecutive anchors.

### Leakage avoidance

The train / val / test split is **strictly contiguous by time** – windows are
never shuffled.  The training set always precedes validation, which always
precedes test.  Because the sliding window only looks backward (into the past
*W* steps) and the label only looks forward (into the future *H* steps), there
is no information flow from the future into training features.

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
| `model`      | Model choice and architecture hyper-parameters     |

## Running Tests

```bash
pytest tests/ -v
```

## License

This project is part of a JetBrains internship application task.