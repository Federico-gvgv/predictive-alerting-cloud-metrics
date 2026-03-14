# Predictive Alerting for Cloud Metrics

> **JetBrains Internship – Task #1**

An early-warning system that learns temporal patterns in cloud-infrastructure
metrics and fires alerts **before** an incident window starts, giving operators
time to react.

---

## Problem Statement

Cloud services emit metric streams (CPU utilisation, memory pressure, network
I/O, …). When a metric crosses a critical threshold (or enters an anomalous
interval), an **incident** can occur — resource exhaustion, latency spikes, or
cascading failures.

Traditional monitoring often alerts **at** the moment of breach. This project
builds a **predictive alerting** pipeline that:

1. Learns normal vs pre-incident patterns from historical metric windows.
2. Outputs a continuous **risk score** at each time step.
3. Converts risk into actionable alerts via an **alert policy** (threshold + cooldown),
   aiming to detect incidents **in advance** with manageable false positives.

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

# Evaluate (pointwise + alerting metrics, plots)
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

For each time step *t* in the metric stream:

| Symbol | Definition | Description |
| ------ | ---------- | ----------- |
| **X_t** | `value[t − W + 1 : t + 1]` | Look-back window of *W* past values (shape `W × K`) |
| **y_t** | `1 if any incident in (t, t + H]` | Binary label: will an incident start within the next *H* steps? |

Defaults: **W = 120** (look-back), **H = 30** (horizon), **stride = 1**.

**Leakage avoidance:** labels scan *strictly future* steps `(t, t+H]` (never the
current step), and splits are time-contiguous (no shuffling).

---

## Dataset

### NAB (default)

The [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) is the primary
data source. NAB is **not vendored** — it is downloaded at runtime via
`scripts/download_nab.py`.

**Default series (single-service setup):**

- `realKnownCause/ambient_temperature_system_failure.csv`

This keeps the project simple and avoids accidental mixing of independent time
series (which would invalidate sliding windows).

**Optional extension (multi-series):**
If enabled in config, each CSV is treated as a separate `series_id` and windows
are created **within** a series (never across series boundaries).

### Synthetic fallback

Set `dataset.source: "synthetic"` for a quick local test without any download.
The generator produces a seeded time series with trend, seasonality, heavy-tailed
noise, regime shifts, and labelled spike incidents.

---

## Models

| Model | Type | Description | Training time |
| ----- | ---- | ----------- | ------------- |
| `heuristic` | Stat. rule | Z-score spike → sigmoid → risk score | — (no training) |
| `logreg` | Linear | Window features → StandardScaler → LogReg | seconds |
| `gbdt` | Ensemble | Same features → GradientBoostingClassifier | seconds |
| `tcn` | Deep | Causal temporal convolutions (raw window input) | ~minutes (CPU) |

### Modeling choices (why these)

- **Heuristic** establishes a training-free floor. Any learning model should beat
  it on meaningful alerting metrics.
- **Logistic regression** is fast, interpretable, and handles imbalance via
  `class_weight="balanced"`.
- **GBDT** is a stronger nonlinear baseline on the same features.
- **TCN** is the main sequence model: it consumes the raw window and can capture
  temporal motifs that handcrafted features miss.

### Why TCN?

- **Causal convolutions:** predictions depend only on the past → avoids future leakage.
- **Dilations:** cover the full receptive field efficiently with few layers.
- **Parallelisable:** faster than sequential RNNs on CPU for this scale.
- **Small footprint:** suitable for laptop-scale experimentation.

---

## Evaluation

### Why alerting metrics matter

Pointwise metrics like ROC-AUC measure ranking quality at every time step, but
they do not answer: *“Would this have warned an operator in time, without
spamming them?”*

This project reports both pointwise and alerting-style metrics:

**Pointwise**

- ROC-AUC, PR-AUC
- Precision/Recall/F1 at the chosen alert threshold

**Alerting-style**

- **Event recall:** % of incidents where at least one alert is raised **before** the incident starts
- **False positives per 10k steps** (or per day/hour if a sampling interval is provided)
- **Lead time:** how early the first alert occurs before incident start (median + IQR)

### Horizon-consistent event matching (important)

Because the task predicts whether an incident occurs **within the next H steps**,
event detection is counted only when the alert happens within a maximum lead time:

- An incident is considered **detected** if the first alert occurs in:
  **[incident_start − H, incident_start)**.
- Lead time is reported in **steps** (and optionally converted to time if the
  sampling interval is known).

This prevents inflated “detections” caused by alerts firing far too early.

### Threshold selection (alert policy)

The alert threshold is selected on the **validation set** by sweeping candidate
thresholds and choosing one that targets ≈ **80% event recall** while minimising
false positives. The chosen threshold is then **frozen** and evaluated on the test set.

### Cooldown / dedup

After an alert fires, further alerts are suppressed for *N* steps (default 10) to
avoid flooding. Results are reported with and without cooldown so the impact is visible.

### Plots

Saved under `outputs/plots/`:

- **PR curve** — precision–recall trade-off
- **Threshold sweep** — event recall vs FP rate across thresholds
- **Lead-time histogram** — distribution of advance warning

---

## Results

Run:

```bash
python -m src.eval --config configs/default.yaml
```

The evaluation prints a compact report and writes plots to `outputs/plots/`.
Results vary by dataset/series and config; the key objective is to demonstrate:

- correct **formulation** (W/H labels),
- sensible **modeling choices** (baselines + sequence model),
- correct **alerting evaluation** (event recall / FP / lead time),
- and clear **analysis** of trade-offs and failure modes.

---

## Analysis of Results (what to look for)

When you push event recall toward ~80%, false positives typically rise — alerting
is a precision–recall trade-off problem. Common patterns:

- **Heuristic** baselines can get decent recall only by firing often → high FP rate.
- **Feature models (logreg/gbdt)** work well when incidents have stable precursors,
  but degrade under regime shifts and heavy-tailed noise.
- **TCN** tends to help when incidents have short temporal motifs that features
  miss, but it needs careful validation and can be sensitive to data volume.

Typical failure modes:

- **False negatives:** abrupt incidents with little warning signal.
- **False positives:** regime shifts, periodic peaks, or heavy-tailed spikes that
  resemble precursors.

---

## Leakage Avoidance

1. **Time-contiguous splits** — train/val/test are consecutive time ranges.
2. **Strictly future labels** — `y_t` scans `(t, t+H]`, never includes the current step.
3. **Causal model** — TCN only sees values ≤ *t*.
4. **Threshold frozen on val** — the test set is never used for model or policy selection.

---

## Production Adaptation (how this becomes a real system)

A realistic alerting system separates **risk scoring** from **alert policy**:

- **Periodic retraining (daily/weekly):** adapt to changing load/infrastructure.
- **Frequent inference (every minute):** score the latest W-step buffer and output risk.
- **Alert policy:** threshold + cooldown + grouping to reduce alert fatigue.
- **Calibration:** calibrate risk scores (Platt/isotonic) so probabilities are meaningful.
- **Drift monitoring:** track input distribution and score drift; trigger retraining when needed.
- **Observability:** log predictions, alerts, and outcomes for continuous improvement.

This aligns with the reference architecture in the internship description (periodic retrain job + frequent inference job).

---

## Repository Structure

```
├── configs/default.yaml        # hyper-parameters (W, H, thresholding, cooldown, model choice)
├── scripts/download_nab.py     # dataset download helper (NAB not vendored)
├── src/
│   ├── train.py                # training entry-point
│   ├── eval.py                 # evaluation entry-point
│   ├── data/                   # loading, windowing, splits
│   ├── models/                 # heuristic, logreg, GBDT, TCN
│   ├── training/               # PyTorch Dataset, training loop (if using TCN)
│   ├── evaluation/             # metrics, thresholding, plots
│   └── utils/                  # config, logging, seed
├── tests/
│   ├── test_labeling.py        # windowing / labeling tests
│   └── test_event_metrics.py   # event metrics / cooldown / horizon constraint tests
├── Makefile
├── requirements.txt
└── README.md
```

> **Scope note.** Only code directly relevant to the predictive-alerting task is included.
> NAB is treated as an external dataset and is not committed to the repository.

---

## License

This project is part of a JetBrains internship application task.
