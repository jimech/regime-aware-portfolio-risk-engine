import json
from pathlib import Path

from regime_risk_engine.research.advanced_demo_workflow import (
    run_advanced_research_demo_workflow,
)


def test_advanced_demo_manifest_points_to_existing_package_files(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    package_dir = result.export_result.output_dir
    manifest = json.loads(result.export_result.manifest_path.read_text())

    memo_path = package_dir / manifest["memo"]
    assert memo_path.exists()
    assert memo_path == result.export_result.memo_path

    for table_filename in manifest["tables"].values():
        table_path = package_dir / table_filename
        assert table_path.exists()
        assert table_path.parent == package_dir


def test_advanced_demo_manifest_includes_all_exported_tables(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    manifest = json.loads(result.export_result.manifest_path.read_text())
    manifest_tables = manifest["tables"]

    assert set(manifest_tables) == set(result.export_result.exported_table_paths)

    for name, path in result.export_result.exported_table_paths.items():
        assert manifest_tables[name] == path.name


def test_advanced_demo_manifest_is_relative_not_absolute(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    manifest = json.loads(result.export_result.manifest_path.read_text())

    assert not Path(manifest["memo"]).is_absolute()

    for table_filename in manifest["tables"].values():
        assert not Path(table_filename).is_absolute()
