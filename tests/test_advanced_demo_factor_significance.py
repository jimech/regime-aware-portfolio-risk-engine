from pathlib import Path

from regime_risk_engine.research.advanced_demo_workflow import (
    run_advanced_research_demo_workflow,
)


def test_advanced_demo_exports_factor_significance_table(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    exported_tables = result.export_result.exported_table_paths

    assert "factor_significance" in exported_tables
    assert exported_tables["factor_significance"].exists()


def test_advanced_demo_memo_includes_factor_significance_section(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    memo = result.export_result.memo_path.read_text(encoding="utf-8")

    assert "## Factor Significance Analysis" in memo
    assert "Regression R-squared" in memo
    assert "P-Value" in memo
