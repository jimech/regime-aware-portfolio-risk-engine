from pathlib import Path

import pytest

from regime_risk_engine.research.advanced_demo_workflow import (
    AdvancedResearchDemoWorkflowError,
    AdvancedResearchDemoWorkflowResult,
    format_advanced_demo_workflow_result,
    run_advanced_research_demo_workflow,
)


def test_run_advanced_research_demo_workflow(tmp_path: Path) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        feature_window=10,
        scenario_horizon=5,
        scenario_simulations=25,
        random_state=7,
        analyst="Jimena Chinchilla",
    )

    assert isinstance(result, AdvancedResearchDemoWorkflowResult)
    assert result.output_dir.exists()
    assert result.input_result.output_dir.exists()
    assert result.export_result.output_dir.exists()
    assert result.export_result.memo_path.exists()

    assert result.input_result.price_data_path.exists()
    assert result.input_result.static_weights_path.exists()
    assert result.input_result.regime_policy_path.exists()
    assert result.input_result.stress_periods_path.exists()
    assert result.input_result.factor_returns_path.exists()

    assert "regime_intelligence_profile" in result.export_result.exported_table_paths
    assert "stress_test_summary" in result.export_result.exported_table_paths
    assert "factor_exposure" in result.export_result.exported_table_paths
    assert "scenario_terminal_summary" in result.export_result.exported_table_paths


def test_format_advanced_demo_workflow_result(tmp_path: Path) -> None:
    result = run_advanced_research_demo_workflow(
        output_dir=tmp_path / "advanced_demo",
        feature_window=10,
        scenario_horizon=5,
        scenario_simulations=25,
        random_state=7,
    )

    output = format_advanced_demo_workflow_result(result)

    assert "Advanced demo workflow completed successfully" in output
    assert "Input directory" in output
    assert "Research package directory" in output
    assert "Advanced memo" in output
    assert "scenario_terminal_summary" in output


def test_run_advanced_demo_rejects_file_output_path(tmp_path: Path) -> None:
    output_path = tmp_path / "not_a_directory.txt"
    output_path.write_text("already exists", encoding="utf-8")

    with pytest.raises(AdvancedResearchDemoWorkflowError, match="not a directory"):
        run_advanced_research_demo_workflow(output_path)


def test_run_advanced_demo_respects_no_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "advanced_demo"

    run_advanced_research_demo_workflow(
        output_dir=output_dir,
        feature_window=10,
        scenario_horizon=5,
        scenario_simulations=25,
        random_state=7,
    )

    with pytest.raises(ValueError, match="overwrite=False"):
        run_advanced_research_demo_workflow(
            output_dir=output_dir,
            feature_window=10,
            scenario_horizon=5,
            scenario_simulations=25,
            random_state=7,
            overwrite=False,
        )
