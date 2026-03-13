#!/usr/bin/env python3
"""Download and extract the NAB (Numenta Anomaly Benchmark) dataset.

Usage::

    python scripts/download_nab.py          # downloads to data/nab/
    python scripts/download_nab.py --dest /path/to/dir

The script downloads the NAB repository as a zip archive from GitHub,
extracts it, and removes the zip.  If the data already exists locally
the download is skipped.

Manual alternative
------------------
If the automatic download fails (e.g. behind a corporate proxy), you
can prepare the data manually:

    1. Download  https://github.com/numenta/NAB/archive/refs/heads/master.zip
    2. Unzip into  data/nab/  so you get  data/nab/NAB-master/data/ ...
    3. Run the training pipeline as usual.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

_NAB_ZIP_URL = "https://github.com/numenta/NAB/archive/refs/heads/master.zip"
_NAB_INNER_PREFIX = "NAB-master"


def download_nab(dest: Path) -> None:
    """Download and extract NAB into *dest*."""
    root = dest / _NAB_INNER_PREFIX
    if root.exists() and (root / "data").is_dir():
        print(f"✓ NAB data already present at {root}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "nab.zip"

    print(f"Downloading NAB from {_NAB_ZIP_URL} …")
    urlretrieve(_NAB_ZIP_URL, zip_path)

    print("Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    zip_path.unlink()
    print(f"✓ NAB extracted to {root}")
    print(f"  CSV data:  {root / 'data'}")
    print(f"  Labels:    {root / 'labels'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the NAB dataset.")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/nab"),
        help="Destination directory (default: data/nab).",
    )
    args = parser.parse_args()

    try:
        download_nab(args.dest)
    except Exception as exc:
        print(f"\n✗ Download failed: {exc}", file=sys.stderr)
        print(
            "\nManual setup:\n"
            "  1. Download https://github.com/numenta/NAB/archive/refs/heads/master.zip\n"
            "  2. Unzip into data/nab/ so you have data/nab/NAB-master/data/\n"
            "  3. Run training as usual.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
