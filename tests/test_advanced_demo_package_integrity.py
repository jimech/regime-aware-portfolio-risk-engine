from pathlib import Path

import pandas as pd

from regime_risk_engine.research.advanced_demo_workflow import (
    run_advanced_research_demo_workflow,
)


def test_advanced_demo_exported_tables_exist_and_are_non_empty(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    exported_tables = result.export_result.exported_table_paths

    assert exported_tables

    for name, path in exported_tables.items():
        assert path.exists(), f"Missing exported table: {name}"
        assert path.stat().st_size > 0, f"Empty exported table file: {name}"

        frame = pd.read_csv(path)
        assert not frame.empty, f"Exported table has no rows: {name}"


def test_advanced_demo_package_contains_expected_research_artifacts(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    package_dir = result.export_result.output_dir
    exported_tables = result.export_result.exported_table_paths

    expected_files = {
        "advanced_research_memo.md",
        "regime_intelligence_profile.csv",
        "stress_test_summary.csv",
        "asset_attribution.csv",
        "factor_exposure.csv",
        "dominant_factor_by_strategy.csv",
        "regime_factor_exposure.csv",
        "rolling_factor_exposures.csv",
        "rolling_factor_exposure_summary.csv",
        "factor_significance.csv",
        "scenario_terminal_summary.csv",
        "scenario_regime_usage.csv",
        "scenario_simulated_paths.csv",
        "scenario_transition_probabilities.csv",
    }

    actual_files = {path.name for path in package_dir.iterdir() if path.is_file()}

    assert expected_files.issubset(actual_files)
    assert set(exported_tables.values()).issubset(set(package_dir.iterdir()))


def test_advanced_demo_memo_references_core_research_sections(
    tmp_path: Path,
) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        analyst="Jimena Chinchilla",
    )

    memo = result.export_result.memo_path.read_text(encoding="utf-8")

    expected_sections = [
        "## Regime Intelligence",
        "## Regime Transition Risk",
        "## Stress-Period Analysis",
        "## Strategy Attribution",
        "## Factor Exposure Analysis",
        "## Rolling Factor Exposure Analysis",
        "## Factor Significance Analysis",
        "## Forward Regime Scenario Simulation",
        "## Research Takeaway",
        "## Advanced Research Limitations",
    ]

    for section in expected_sections:
        assert section in memo
