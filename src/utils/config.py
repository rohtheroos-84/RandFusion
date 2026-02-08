"""
Configuration loader for RandFusion.

Reads configs/config.yaml and provides a single dict accessible
throughout the project. All tunable parameters live in that file.
"""

import os
from pathlib import Path
import yaml


# Project root = two levels up from this file (src/utils/config.py → RandFusion/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load YAML configuration file and return as a dictionary.

    Args:
        config_path: Path to config file. Defaults to configs/config.yaml.

    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# Singleton: loaded once, imported everywhere
_config = None


def get_config() -> dict:
    """Return the cached configuration (loads on first call)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
