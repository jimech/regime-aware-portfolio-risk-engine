import json
from pathlib import Path

import pytest

from regime_risk_engine.cli import main
from regime_risk_engine.cli_config import (
    CliConfigInspectionError,
    ConfigInspectionResult,
    format_config_inspection,
    inspect_config_file,
)


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        """
data:
  start_date: "2015-01-01"
  end_date: "2024-12-31"
paths:
  data_dir: "data"
  report_dir: "reports"
universe:
  assets:
    - ticker: SPY
      asset_class: equity
    - ticker: TLT
      asset_class: bond
portfolio:
  static_weights:
    SPY: 0.6
    TLT: 0.4
""",
        encoding="utf-8",
    )

    return config_path


def test_inspect_config_file(tmp_path: Path) -> None:
    result = inspect_config_file(write_config(tmp_path))

    assert isinstance(result, ConfigInspectionResult)
    assert result.ticker_count == 2
    assert result.tickers == ["SPY", "TLT"]
    assert result.start_date == "2015-01-01"
    assert result.end_date == "2024-12-31"
    assert result.data_directory == "data"
    assert result.report_directory == "reports"
    assert result.top_level_keys == ["data", "paths", "portfolio", "universe"]


def test_config_inspection_result_to_dict(tmp_path: Path) -> None:
    result = inspect_config_file(write_config(tmp_path))
    result_dict = result.to_dict()

    assert result_dict["ticker_count"] == 2
    assert result_dict["tickers"] == ["SPY", "TLT"]
    assert result_dict["config_path"] == str(write_config(tmp_path).resolve())


def test_format_config_inspection(tmp_path: Path) -> None:
    result = inspect_config_file(write_config(tmp_path))
    formatted = format_config_inspection(result)

    assert "Config inspection" in formatted
    assert "ticker_count: 2" in formatted
    assert "SPY, TLT" in formatted


def test_cli_inspect_config_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path)

    exit_code = main(["inspect-config", "--config", str(config_path)])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Config inspection" in captured.out
    assert "ticker_count: 2" in captured.out


def test_cli_inspect_config_json_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path)

    exit_code = main(
        [
            "inspect-config",
            "--config",
            str(config_path),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert exit_code == 0
    assert result["ticker_count"] == 2
    assert result["tickers"] == ["SPY", "TLT"]


def test_inspect_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CliConfigInspectionError, match="does not exist"):
        inspect_config_file(tmp_path / "missing.yaml")


def test_inspect_config_rejects_non_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "base.txt"
    config_path.write_text("key: value", encoding="utf-8")

    with pytest.raises(CliConfigInspectionError, match="must be YAML"):
        inspect_config_file(config_path)


def test_inspect_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text("- SPY\n- TLT\n", encoding="utf-8")

    with pytest.raises(CliConfigInspectionError, match="mapping"):
        inspect_config_file(config_path)


def test_cli_inspect_config_rejects_missing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "inspect-config",
                "--config",
                str(tmp_path / "missing.yaml"),
            ]
        )

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "does not exist" in captured.err
