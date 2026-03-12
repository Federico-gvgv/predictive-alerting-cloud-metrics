"""NAB (Numenta Anomaly Benchmark) dataset downloader and loader.

The module can:

1. **Auto-download** the NAB repository as a zip archive and extract it
   into a local directory (default ``data/nab``).
2. **Parse** one or more CSV files from a chosen subset (e.g.
   ``realKnownCause``) together with the corresponding anomaly-window
   labels from ``labels/combined_windows.json``.
3. Return a **unified DataFrame** with columns
   ``["timestamp", "value", "is_incident"]``.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

_NAB_ZIP_URL = (
    "https://github.com/numenta/NAB/archive/refs/heads/master.zip"
)
_NAB_INNER_PREFIX = "NAB-master"  # top-level dir inside the zip


# ── Download / extract ──────────────────────────────────────────────────


def download_nab(dest: str | Path) -> Path:
    """Download and extract the NAB repository into *dest*.

    If *dest* already contains the extracted data the download is skipped.

    Parameters
    ----------
    dest:
        Target directory (e.g. ``data/nab``).

    Returns
    -------
    Path
        Root of the extracted NAB tree (``dest / NAB-master``).
    """
    dest = Path(dest)
    root = dest / _NAB_INNER_PREFIX

    if root.exists() and (root / "data").is_dir():
        logger.info("NAB data already present at %s – skipping download.", root)
        return root

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "nab.zip"

    logger.info("Downloading NAB from %s …", _NAB_ZIP_URL)
    urlretrieve(_NAB_ZIP_URL, zip_path)
    logger.info("Download complete – extracting …")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    zip_path.unlink()  # clean up the zip
    logger.info("NAB extracted to %s.", root)
    return root


# ── Label parsing ────────────────────────────────────────────────────────


def _load_anomaly_windows(nab_root: Path) -> dict[str, list[tuple[str, str]]]:
    """Load ``labels/combined_windows.json`` and return a mapping.

    Returns
    -------
    dict
        ``{relative_csv_path: [(start_ts, end_ts), …]}``
    """
    windows_file = nab_root / "labels" / "combined_windows.json"
    if not windows_file.is_file():
        raise FileNotFoundError(
            f"Anomaly-window labels not found at {windows_file}"
        )
    with windows_file.open("r", encoding="utf-8") as fh:
        raw: dict[str, list[list[str]]] = json.load(fh)

    # Convert [[start, end], …] → [(start, end), …]
    return {k: [(s, e) for s, e in v] for k, v in raw.items()}


def _mark_incidents(
    df: pd.DataFrame,
    windows: list[tuple[str, str]],
) -> pd.Series:
    """Return a boolean Series marking rows inside any anomaly window."""
    mask = pd.Series(False, index=df.index)
    for start, end in windows:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask |= (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
    return mask


# ── Public loader ────────────────────────────────────────────────────────


def load_nab(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load NAB CSVs for the configured subset and return a unified DataFrame.

    Parameters
    ----------
    cfg:
        Full experiment config.  Relevant keys under ``cfg["dataset"]["nab"]``:

        * ``data_dir``       – local path for the NAB tree (default ``data/nab``).
        * ``subset``         – subdirectory under ``data/`` (default
          ``realKnownCause``).
        * ``files``          – explicit list of filenames, or ``null`` for all.
        * ``auto_download``  – whether to auto-download if missing (default True).

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``value``, ``is_incident``.
    """
    nab_cfg: dict[str, Any] = cfg["dataset"].get("nab", {})
    data_dir = Path(nab_cfg.get("data_dir", "data/nab"))
    subset: str = nab_cfg.get("subset", "realKnownCause")
    file_filter: list[str] | None = nab_cfg.get("files", None)
    auto_download: bool = nab_cfg.get("auto_download", True)

    # Download if needed
    nab_root = data_dir / _NAB_INNER_PREFIX
    if auto_download and not (nab_root / "data").is_dir():
        nab_root = download_nab(data_dir)

    if not (nab_root / "data").is_dir():
        raise FileNotFoundError(
            f"NAB data directory not found at {nab_root / 'data'}. "
            "Set auto_download: true or download manually."
        )

    # Locate CSV files for the chosen subset
    subset_dir = nab_root / "data" / subset
    if not subset_dir.is_dir():
        available = sorted(p.name for p in (nab_root / "data").iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"Subset '{subset}' not found. Available: {available}"
        )

    csv_files = sorted(subset_dir.glob("*.csv"))
    if file_filter:
        csv_files = [f for f in csv_files if f.name in file_filter]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {subset_dir}")

    logger.info("Loading %d NAB CSV(s) from subset '%s'.", len(csv_files), subset)

    # Load anomaly windows
    all_windows = _load_anomaly_windows(nab_root)

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df.rename(columns={"timestamp": "timestamp", "value": "value"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[["timestamp", "value"]].copy()

        # Find matching anomaly windows.  Keys in combined_windows.json look
        # like "realKnownCause/ambient_temperature_system_failure.csv".
        label_key = f"{subset}/{csv_path.name}"
        windows = all_windows.get(label_key, [])
        df["is_incident"] = _mark_incidents(df, windows)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "NAB loaded – %d rows, %d incident rows (%.2f%%).",
        len(result),
        result["is_incident"].sum(),
        result["is_incident"].mean() * 100,
    )
    return result
