import json
from pathlib import Path

from regime_risk_engine.research.advanced_demo_workflow import (
    run_advanced_research_demo_workflow,
)


def test_advanced_demo_package_writes_manifest(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    manifest_path = result.export_result.manifest_path

    assert manifest_path.exists()
    assert manifest_path.name == "manifest.json"


def test_advanced_demo_manifest_references_memo_and_tables(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    manifest = json.loads(result.export_result.manifest_path.read_text())

    assert manifest["memo"] == "advanced_research_memo.md"
    assert manifest["tables"]

    for name, path in result.export_result.exported_table_paths.items():
        assert manifest["tables"][name] == path.name


def test_advanced_demo_json_output_includes_manifest_path(
    tmp_path: Path,
    capsys,
) -> None:
    from regime_risk_engine.cli import main

    exit_code = main(
        [
            "run-advanced-demo",
            "--output-dir",
            str(tmp_path / "advanced_demo"),
            "--analyst",
            "Jimena Chinchilla",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["manifest_path"].endswith("manifest.json")
