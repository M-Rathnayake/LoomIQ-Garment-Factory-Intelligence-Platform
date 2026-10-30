"""Project configuration.

Loads settings from ``configs/config.yaml`` and secrets from ``.env``.

Example:
    >>> from loomiq.config import load_config, get_path
    >>> cfg = load_config()
    >>> cfg["project"]["random_seed"]
    42
    >>> get_path("raw_data")  # absolute path to data/raw
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# src/loomiq/config.py -> parents[2] is the project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"
ENV_PATH: Path = PROJECT_ROOT / ".env"

# Load .env once on import. Real environment variables take priority.
load_dotenv(ENV_PATH, override=False)


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config file and return it as a dictionary.

    Args:
        path: Location of the YAML file. Defaults to ``configs/config.yaml``.

    Returns:
        The parsed configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_path(key: str) -> Path:
    """Return an absolute project path from the ``paths`` section of the config.

    Args:
        key: A key under ``paths`` in config.yaml, e.g. ``"raw_data"``.

    Returns:
        Absolute path resolved against the project root.
    """
    relative = load_config()["paths"][key]
    return PROJECT_ROOT / relative


def get_env(name: str, default: str | None = None) -> str | None:
    """Return an environment variable (from the shell or ``.env``).

    Args:
        name: Variable name, e.g. ``"ANTHROPIC_API_KEY"``.
        default: Value to return if the variable is not set or empty.
    """
    return os.getenv(name) or default
