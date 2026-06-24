from pathlib import Path

from regime_risk_engine.research.advanced_demo_workflow import (
    run_advanced_research_demo_workflow,
)


def test_advanced_demo_exports_rolling_factor_exposure_tables(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    exported_tables = result.export_result.exported_table_paths

    assert "rolling_factor_exposures" in exported_tables
    assert "rolling_factor_exposure_summary" in exported_tables
    assert exported_tables["rolling_factor_exposures"].exists()
    assert exported_tables["rolling_factor_exposure_summary"].exists()


def test_advanced_demo_memo_includes_rolling_factor_exposure_section(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    memo = result.export_result.memo_path.read_text(encoding="utf-8")

    assert "## Rolling Factor Exposure Analysis" in memo
    assert "factor betas changed through time" in memo
