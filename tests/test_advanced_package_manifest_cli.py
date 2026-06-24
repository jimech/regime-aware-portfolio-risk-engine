import json
from pathlib import Path

from regime_risk_engine.cli import main
from regime_risk_engine.research.advanced_demo_workflow import (
    run_advanced_research_demo_workflow,
)


def test_cli_inspect_advanced_package(
    tmp_path: Path,
    capsys,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    exit_code = main(
        [
            "inspect-advanced-package",
            "--package-dir",
            str(result.export_result.output_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Advanced research package inspection" in captured.out
    assert "manifest.json" in captured.out
    assert "advanced_research_memo.md" in captured.out


def test_cli_inspect_advanced_package_json(
    tmp_path: Path,
    capsys,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    exit_code = main(
        [
            "inspect-advanced-package",
            "--package-dir",
            str(result.export_result.output_dir),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["manifest_path"].endswith("manifest.json")
    assert payload["memo_path"].endswith("advanced_research_memo.md")
    assert "factor_significance" in payload["tables"]


def test_cli_inspect_advanced_package_returns_error_for_missing_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()

    exit_code = main(
        [
            "inspect-advanced-package",
            "--package-dir",
            str(package_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Advanced package inspection failed" in captured.out
