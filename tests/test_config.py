from pathlib import Path

import pytest
import yaml

from regime_risk_engine.config import ConfigError, load_config, validate_config


def test_load_config_success(tmp_path: Path) -> None:
    config_data = {
        "project": {},
        "paths": {},
        "data": {},
        "assets": {},
        "portfolio": {},
        "features": {},
        "regime_model": {},
        "backtest": {},
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    config = load_config(config_path)

    assert config == config_data


def test_load_config_missing_file_raises_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match="Config file does not exist"):
        load_config(missing_path)


def test_validate_config_missing_required_section_raises_error() -> None:
    config_data = {
        "project": {},
        "paths": {},
    }

    with pytest.raises(ConfigError, match="Missing required config section"):
        validate_config(config_data)
