# Predictive Alerting for Cloud Metrics

> **JetBrains Internship – Task #1**

Early-warning system that learns temporal patterns in cloud-infrastructure
metrics and fires alerts **before** a threshold is breached, giving operators
time to react.

## Problem Statement

Cloud platforms continuously emit metric streams (CPU utilisation, memory
pressure, network I/O, …).  When a metric crosses a critical threshold an
**incident** occurs — resource exhaustion, latency spikes, or cascading
failures.

Traditional monitoring fires alerts *at the moment of breach*, leaving
no time for mitigation.  This project builds a **predictive alerting**
pipeline that:

1. Learns normal vs pre-incident patterns from historical metric windows.
2. Outputs a continuous risk score at each time-step.
3. Fires alerts minutes-to-hours *before* the incident starts.

---

## Quick Start

```bash
# Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: make install

# Download the NAB dataset (≈ 60 MB, one-time)
python scripts/download_nab.py           # or: make data

# Train (logistic regression by default — seconds on CPU)
python -m src.train --config configs/default.yaml    # or: make train

# Evaluate (pointwise + event metrics, plots)
python -m src.eval --config configs/default.yaml     # or: make eval

# Run tests
pytest tests/ -v                         # or: make test
```

To train the **TCN** instead:

```yaml
# configs/default.yaml
model:
  model_choice: "tcn"    # heuristic | logreg | gbdt | tcn
```

---

## Sliding-Window Formulation

For each time-step *t* in the metric stream:

| Symbol | Definition | Description |
| ------ | ---------- | ----------- |
| **X_t** | `value[t − W + 1 : t + 1]` | Look-back window of *W* past values |
| **y_t** | `1 if any incident in (t, t + H]` | Binary label: will an incident start in the next *H* steps? |

Defaults: **W = 120** (look-back), **H = 30** (forecast horizon), **stride = 1**.

The label scans *strictly future* steps — the value at *t* itself is never
part of the label.  Combined with time-contiguous train/val/test splits
(no shuffling), this prevents data leakage.

---

## Dataset

### NAB (default)

The [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) is the
primary data source.  NAB is **not vendored** — it is downloaded at runtime
via `scripts/download_nab.py` or automatically on first training run.

The default subset (`realKnownCause`, 7 CSV files, ~70 k rows) keeps CPU
runtime well under 10 minutes.

### Synthetic fallback

Set `dataset.source: "synthetic"` for a quick local test without any
download.  The generator produces a seeded time series with trend,
seasonality, heavy-tailed noise, regime shifts, and labelled spike
incidents.

---

## Models

| Model | Type | Description | Training time |
| ----- | ---- | ----------- | ------------- |
| `heuristic` | Stat. rule | Z-score of last value → sigmoid → risk score | — (no training) |
| `logreg` | Linear | 12 hand-crafted features → StandardScaler → LogReg | seconds |
| `gbdt` | Ensemble | Same features → GradientBoostingClassifier | seconds |
| `tcn` | Deep | Causal temporal convolutions (raw window input) | 1–5 min (CPU) |

### Why these models?

- **Heuristic** establishes a training-free floor.  Any learning model
  should beat it.
- **Logistic regression** is fast, interpretable, and handles class
  imbalance via `class_weight="balanced"`.  It proves that even simple
  features carry predictive signal.
- **TCN** (Temporal Convolutional Network) is the main sequence model.

### Why TCN?

| Property | Benefit |
| -------- | ------- |
| Causal convolutions | Each position depends only on the past — no future leakage |
| Exponential dilation (`2^i`) | Covers the full W-step receptive field with only 3 layers |
| Parallelisable | Convolutions run much faster on CPU than sequential RNNs |
| Small footprint | Default config: ~36 k parameters — trains on a laptop |

Limitations: univariate only (single `value` channel), no attention, no
long-range dependencies beyond the receptive field.

---

## Evaluation

### Why alerting metrics matter

ROC-AUC measures ranking quality at every time-step.  It does **not** tell
you whether the system would have saved an operator from an outage.  For
that you need:

| Metric | What it captures |
| ------ | ---------------- |
| **Event recall** | % of incidents detected *before* they start |
| **FP per 10 k steps** | Alert noise — too high and operators ignore the system |
| **Lead time** (median, IQR) | How much advance warning operators get |

The project reports **both** pointwise (ROC-AUC, PR-AUC, P/R/F1) and
event-level metrics.

### Threshold selection

During training, the alert threshold is selected on the **validation set**
by sweeping 100 candidates and picking the one targeting ≈ 80 % event
recall with the fewest false positives.  The threshold is frozen and
applied to the test set.

### Cooldown / dedup

After an alert fires, further alerts are suppressed for *N* steps
(default 10) to avoid flooding.  Results are reported with and without
cooldown so the impact is visible.

### Plots

Saved under `outputs/plots/`:

- **PR curve** — precision–recall trade-off
- **Lead-time histogram** — distribution of advance warning
- **Threshold sweep** — event recall vs FP rate across thresholds

---

## Example Results

> Results on NAB `realKnownCause` with default config.

| Model | ROC-AUC | PR-AUC | Event Recall | FP / 10 k | Lead Time (median) |
| ----- | ------- | ------ | ------------ | --------- | ------------------- |
| `heuristic` | 0.67 | 0.39 | 1.00 | 864 | ~ 38 h |
| `logreg` | 0.57 | 0.13 | 1.00 | 789 | — |
| `tcn` | — | — | — | — | — |

*(TCN results depend on training — fill in after running `make train` with
`model_choice: "tcn"`.)*

---

## Leakage Avoidance

1. **Time-contiguous splits** — train, val, test are consecutive time
   ranges; no shuffling.
2. **Strictly future labels** — `y_t` scans `(t, t+H]`, never includes
   the current step.
3. **Causal model** — TCN padding is "chomped" so position *t* only sees
   values ≤ *t*.
4. **Threshold frozen on val** — the test set is never used for model or
   threshold selection.

---

## Production Adaptation

If this were deployed in a production monitoring pipeline:

| Concern | Approach |
| ------- | -------- |
| **Periodic retraining** | Schedule a nightly or weekly re-train on the latest metric window to adapt to infrastructure changes |
| **Frequent inference** | Run the model every 1–5 min on the latest W-step buffer; serve via a lightweight REST endpoint or sidecar |
| **Calibration** | Apply Platt scaling or isotonic regression so that the output probability is meaningful for SLA-based policies |
| **Drift monitoring** | Track the distribution of risk scores over time; alert on distributional shift (KS test, PSI) |
| **Alert policy & dedup** | Combine cooldown with exponential back-off; integrate with PagerDuty / OpsGenie; group correlated alerts |

---

## Repository Structure

```
├── configs/default.yaml        # all hyper-parameters
├── scripts/download_nab.py     # dataset download helper
├── src/
│   ├── train.py                # training entry-point
│   ├── eval.py                 # evaluation entry-point
│   ├── data/                   # loading, windowing, splits
│   ├── models/                 # heuristic, logreg, GBDT, TCN
│   ├── training/               # PyTorch Dataset, training loop
│   ├── evaluation/             # metrics, thresholding, plots
│   └── utils/                  # config, logging, seed
├── tests/
│   ├── test_labeling.py        # windowing / labeling tests
│   └── test_event_metrics.py   # event metrics / cooldown tests
├── Makefile
├── requirements.txt
└── README.md
```

> **Scope note.**  Only code directly relevant to the predictive-alerting
> task is included.  NAB is treated as an external dataset.

## Configuration

All hyper-parameters live in [`configs/default.yaml`](configs/default.yaml).

## License

This project is part of a JetBrains internship application task.