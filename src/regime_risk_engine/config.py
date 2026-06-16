from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL_SECTIONS = {
    "project",
    "paths",
    "data",
    "assets",
    "portfolio",
    "features",
    "regime_model",
    "backtest",
}


class ConfigError(Exception):
    """Raised when project configuration is invalid."""


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML project configuration file.

    Args:
        config_path: Path to a YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigError: If the config file is missing, empty, or invalid.
    """
    path = Path(config_path)

    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ConfigError(f"Config file is empty or invalid: {path}")

    validate_config(config)

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required top-level config sections.

    Args:
        config: Parsed configuration dictionary.

    Raises:
        ConfigError: If required sections are missing.
    """
    missing_sections = REQUIRED_TOP_LEVEL_SECTIONS.difference(config)

    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ConfigError(f"Missing required config section(s): {missing}")
