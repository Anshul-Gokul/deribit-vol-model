"""Configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_config_cache: dict[str, Any] | None = None


def get_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file, with caching."""
    global _config_cache
    if _config_cache is not None and config_path is None:
        return _config_cache

    if config_path is None:
        config_path = Path("configs/default.yaml")
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    # Override with environment variables
    if os.getenv("DERIBIT_USE_TESTNET", "").lower() == "true":
        cfg.setdefault("data", {})["use_testnet"] = True

    if config_path == Path("configs/default.yaml"):
        _config_cache = cfg

    return cfg


def get_data_dir() -> Path:
    """Get the data directory from env or default."""
    return Path(os.getenv("DATA_DIR", "data"))
