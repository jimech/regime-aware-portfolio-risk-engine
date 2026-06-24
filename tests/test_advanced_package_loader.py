from pathlib import Path

import pandas as pd

from regime_risk_engine.research.advanced_demo_workflow import (
    run_advanced_research_demo_workflow,
)
from regime_risk_engine.research.package_manifest import (
    load_advanced_research_package,
)


def test_load_advanced_research_package_reads_memo_and_tables(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    package = load_advanced_research_package(result.export_result.output_dir)

    assert package.package_dir == result.export_result.output_dir
    assert package.manifest_path == result.export_result.manifest_path
    assert "Advanced Regime-Aware Portfolio Research Memo" in package.memo
    assert package.tables


def test_load_advanced_research_package_loads_expected_tables(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    package = load_advanced_research_package(result.export_result.output_dir)

    assert "factor_significance" in package.tables
    assert "rolling_factor_exposures" in package.tables
    assert "scenario_terminal_summary" in package.tables

    for table in package.tables.values():
        assert isinstance(table, pd.DataFrame)
        assert not table.empty


def test_advanced_research_package_table_names_are_sorted(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    package = load_advanced_research_package(result.export_result.output_dir)

    assert package.table_names() == sorted(package.tables)
