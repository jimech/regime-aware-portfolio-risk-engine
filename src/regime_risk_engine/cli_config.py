from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class CliConfigInspectionError(ValueError):
    """Raised when a config file cannot be inspected from the CLI."""


@dataclass(frozen=True, slots=True)
class ConfigInspectionResult:
    """Summary of a project config file."""

    config_path: Path
    top_level_keys: list[str]
    ticker_count: int
    tickers: list[str]
    start_date: str | None
    end_date: str | None
    data_directory: str | None
    report_directory: str | None

    def to_dict(self) -> dict[str, object]:
        """Convert the config inspection result to a JSON-safe dictionary."""
        return {
            "config_path": str(self.config_path),
            "top_level_keys": self.top_level_keys,
            "ticker_count": self.ticker_count,
            "tickers": self.tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "data_directory": self.data_directory,
            "report_directory": self.report_directory,
        }


def inspect_config_file(config_path: Path | str) -> ConfigInspectionResult:
    """Inspect a YAML config file and return a CLI-friendly summary."""
    clean_config_path = Path(config_path).expanduser().resolve()
    config = _load_yaml_mapping(clean_config_path)

    tickers = _extract_tickers(config)

    return ConfigInspectionResult(
        config_path=clean_config_path,
        top_level_keys=sorted(str(key) for key in config),
        ticker_count=len(tickers),
        tickers=tickers,
        start_date=_extract_first_string(
            config,
            [
                ("data", "start_date"),
                ("market_data", "start_date"),
                ("backtest", "start_date"),
                ("start_date",),
            ],
        ),
        end_date=_extract_first_string(
            config,
            [
                ("data", "end_date"),
                ("market_data", "end_date"),
                ("backtest", "end_date"),
                ("end_date",),
            ],
        ),
        data_directory=_extract_first_string(
            config,
            [
                ("paths", "data_dir"),
                ("paths", "data_directory"),
                ("data", "data_dir"),
                ("data", "output_dir"),
                ("data_dir",),
            ],
        ),
        report_directory=_extract_first_string(
            config,
            [
                ("paths", "report_dir"),
                ("paths", "report_directory"),
                ("reporting", "output_dir"),
                ("reports", "output_dir"),
                ("report_dir",),
            ],
        ),
    )


def format_config_inspection(result: ConfigInspectionResult) -> str:
    """Format a config inspection result for terminal output."""
    lines = [
        "Config inspection",
        f"- config_path: {result.config_path}",
        f"- top_level_keys: {', '.join(result.top_level_keys)}",
        f"- ticker_count: {result.ticker_count}",
    ]

    if result.tickers:
        lines.append(f"- tickers: {', '.join(result.tickers)}")
    else:
        lines.append("- tickers: none detected")

    lines.extend(
        [
            f"- start_date: {result.start_date or 'not detected'}",
            f"- end_date: {result.end_date or 'not detected'}",
            f"- data_directory: {result.data_directory or 'not detected'}",
            f"- report_directory: {result.report_directory or 'not detected'}",
        ]
    )

    return "\n".join(lines)


def _load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise CliConfigInspectionError(f"Config file does not exist: {config_path}")

    if not config_path.is_file():
        raise CliConfigInspectionError(f"Config path is not a file: {config_path}")

    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise CliConfigInspectionError(f"Config file must be YAML: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        loaded_config = yaml.safe_load(file)

    if not isinstance(loaded_config, dict):
        raise CliConfigInspectionError("Config YAML must contain a mapping/object")

    return loaded_config


def _extract_tickers(config: dict[str, Any]) -> list[str]:
    candidate_values = [
        _get_nested_value(config, ("universe",)),
        _get_nested_value(config, ("universe", "assets")),
        _get_nested_value(config, ("assets",)),
        _get_nested_value(config, ("data", "tickers")),
        _get_nested_value(config, ("market_data", "tickers")),
        _get_nested_value(config, ("tickers",)),
        _get_nested_value(config, ("static_weights",)),
        _get_nested_value(config, ("portfolio", "static_weights")),
    ]

    tickers: list[str] = []

    for value in candidate_values:
        tickers.extend(_extract_tickers_from_value(value))

    return sorted(set(tickers))


def _extract_tickers_from_value(value: object) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [_clean_ticker(value)]

    if isinstance(value, list):
        tickers: list[str] = []

        for item in value:
            tickers.extend(_extract_tickers_from_value(item))

        return tickers

    if isinstance(value, dict):
        if "ticker" in value:
            ticker_value = value["ticker"]

            if isinstance(ticker_value, str):
                return [_clean_ticker(ticker_value)]

        if "tickers" in value:
            return _extract_tickers_from_value(value["tickers"])

        if "assets" in value:
            return _extract_tickers_from_value(value["assets"])

        if value and all(_looks_numeric(item) for item in value.values()):
            return [_clean_ticker(str(key)) for key in value]

        tickers = []

        for nested_value in value.values():
            tickers.extend(_extract_tickers_from_value(nested_value))

        return tickers

    return []


def _extract_first_string(
    config: dict[str, Any],
    paths: list[tuple[str, ...]],
) -> str | None:
    for path in paths:
        value = _get_nested_value(config, path)

        if value is None:
            continue

        clean_value = str(value).strip()

        if clean_value:
            return clean_value

    return None


def _get_nested_value(config: dict[str, Any], path: tuple[str, ...]) -> object:
    current_value: object = config

    for key in path:
        if not isinstance(current_value, dict):
            return None

        current_value = current_value.get(key)

    return current_value


def _clean_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _looks_numeric(value: object) -> bool:
    if isinstance(value, bool):
        return False

    if isinstance(value, int | float):
        return True

    try:
        float(str(value))
    except ValueError:
        return False

    return True
