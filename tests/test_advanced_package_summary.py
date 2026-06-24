from pathlib import Path

from regime_risk_engine.research.advanced_demo_workflow import (
    run_advanced_research_demo_workflow,
)
from regime_risk_engine.research.package_manifest import (
    load_advanced_research_package,
    summarize_advanced_research_package,
)


def test_summarize_advanced_research_package_builds_dashboard_summary(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )
    package = load_advanced_research_package(result.export_result.output_dir)

    summary = summarize_advanced_research_package(package)

    assert summary.memo_title == "Advanced Regime-Aware Portfolio Research Memo"
    assert summary.table_count == len(package.tables)
    assert summary.table_names == sorted(package.tables)
    assert summary.has_factor_significance is True
    assert summary.has_rolling_factor_exposure is True
    assert summary.has_scenario_simulation is True
    assert summary.has_stress_test is True


def test_advanced_research_package_summary_to_dict(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )
    package = load_advanced_research_package(result.export_result.output_dir)

    payload = summarize_advanced_research_package(package).to_dict()

    assert payload["memo_title"] == "Advanced Regime-Aware Portfolio Research Memo"
    assert payload["table_count"] == len(package.tables)
    assert payload["has_factor_significance"] is True
    assert payload["has_rolling_factor_exposure"] is True
