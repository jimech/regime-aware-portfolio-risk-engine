import json
from pathlib import Path

import pytest

from regime_risk_engine.research.advanced_demo_workflow import (
    run_advanced_research_demo_workflow,
)
from regime_risk_engine.research.package_manifest import (
    AdvancedPackageManifestError,
    format_advanced_package_manifest_inspection,
    inspect_advanced_package_manifest,
)


def test_inspect_advanced_package_manifest_reads_generated_package(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    inspection = inspect_advanced_package_manifest(result.export_result.output_dir)

    assert inspection.manifest_path.name == "manifest.json"
    assert inspection.memo_path == result.export_result.memo_path
    assert set(inspection.table_paths) == set(result.export_result.exported_table_paths)


def test_format_advanced_package_manifest_inspection(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    inspection = inspect_advanced_package_manifest(result.export_result.output_dir)
    formatted = format_advanced_package_manifest_inspection(inspection)

    assert "Advanced research package inspection" in formatted
    assert "advanced_research_memo.md" in formatted
    assert "factor_significance" in formatted
    assert "rolling_factor_exposures" in formatted


def test_inspect_advanced_package_manifest_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()

    with pytest.raises(AdvancedPackageManifestError, match="Manifest file"):
        inspect_advanced_package_manifest(package_dir)


def test_inspect_advanced_package_manifest_rejects_missing_table(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "advanced_research_memo.md").write_text("# Memo\n", encoding="utf-8")
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "memo": "advanced_research_memo.md",
                "tables": {"missing": "missing.csv"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AdvancedPackageManifestError, match="does not exist"):
        inspect_advanced_package_manifest(package_dir)
