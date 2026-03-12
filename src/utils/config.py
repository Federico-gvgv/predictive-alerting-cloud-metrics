"""YAML configuration loader.

Provides a single ``load_config`` helper that reads a YAML file, merges it
with sensible defaults produced by the experiment schema, and returns the
result as a plain dictionary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary.

    Parameters
    ----------
    path:
        Filesystem path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    ValueError
        If the YAML content is empty or not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError(
            f"Expected a YAML mapping at the top level, got {type(cfg).__name__}"
        )

    return cfg


def pretty_print_config(cfg: dict[str, Any], indent: int = 2) -> str:
    """Return a human-readable YAML representation of *cfg*."""
    return yaml.dump(cfg, default_flow_style=False, indent=indent, sort_keys=False)
