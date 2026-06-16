import json
from pathlib import Path

import pytest

from regime_risk_engine import __version__
from regime_risk_engine.cli import (
    CliError,
    build_healthcheck_result,
    build_parser,
    main,
)


def test_build_parser() -> None:
    parser = build_parser()

    assert parser.prog == "regime-risk-engine"


def test_cli_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["version"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == __version__


def test_cli_healthcheck_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["healthcheck"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Regime Risk Engine healthcheck" in captured.out
    assert "import_ok: True" in captured.out


def test_cli_healthcheck_json_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["healthcheck", "--json"])

    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert exit_code == 0
    assert result["package_version"] == __version__
    assert result["import_ok"] is True


def test_cli_healthcheck_creates_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"

    exit_code = main(["healthcheck", "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert output_dir.exists()
    assert output_dir.is_dir()


def test_build_healthcheck_result() -> None:
    result = build_healthcheck_result()

    assert result["package_version"] == __version__
    assert result["import_ok"] is True
    assert result["output_dir"] is None


def test_build_healthcheck_result_creates_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"

    result = build_healthcheck_result(output_dir=output_dir)

    assert output_dir.exists()
    assert result["output_dir"] == str(output_dir.resolve())
    assert result["output_dir_exists"] is True


def test_healthcheck_rejects_output_path_that_is_file(tmp_path: Path) -> None:
    output_file = tmp_path / "not_a_directory"
    output_file.write_text("content", encoding="utf-8")

    with pytest.raises(CliError, match="not a directory"):
        build_healthcheck_result(output_dir=output_file)


def test_cli_rejects_missing_command() -> None:
    with pytest.raises(SystemExit):
        main([])
