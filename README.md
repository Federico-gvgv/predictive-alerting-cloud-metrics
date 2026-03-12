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

### 3. Run the training stub

```bash
python -m src.train --config configs/default.yaml
```

### 4. Run the evaluation stub

```bash
python -m src.eval --config configs/default.yaml
```

Both commands will parse the config, set a deterministic seed, configure
logging, and print the loaded configuration together with next-step TODOs.

## Configuration

All hyper-parameters live in `configs/default.yaml`. Key sections:

| Section      | Purpose                                            |
| ------------ | -------------------------------------------------- |
| `dataset`    | Data source, target metric, resampling interval    |
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