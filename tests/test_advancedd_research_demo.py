from pathlib import Path

import pandas as pd
import pytest

from regime_risk_engine.research.advanced_cli_export import (
    export_advanced_research_from_files,
)
from regime_risk_engine.research.advanced_demo import (
    AdvancedResearchDemoInputError,
    AdvancedResearchDemoInputResult,
    create_advanced_research_demo_inputs,
    format_advanced_demo_input_result,
)


def test_create_advanced_research_demo_inputs(tmp_path: Path) -> None:
    result = create_advanced_research_demo_inputs(tmp_path / "demo_inputs")

    assert isinstance(result, AdvancedResearchDemoInputResult)
    assert result.output_dir.exists()
    assert result.price_data_path.exists()
    assert result.static_weights_path.exists()
    assert result.regime_policy_path.exists()
    assert result.stress_periods_path.exists()
    assert result.factor_returns_path.exists()

    output = format_advanced_demo_input_result(result)

    assert "Advanced research demo inputs created successfully" in output
    assert "prices.csv" in output


def test_demo_price_data_has_expected_shape_and_columns(tmp_path: Path) -> None:
    result = create_advanced_research_demo_inputs(tmp_path / "demo_inputs")

    price_data = pd.read_csv(result.price_data_path)

    assert set(price_data.columns) == {"date", "ticker", "adjusted_close"}
    assert set(price_data["ticker"]) == {"SPY", "TLT", "GLD"}
    assert len(price_data) == 540


def test_demo_static_weights_sum_to_one(tmp_path: Path) -> None:
    result = create_advanced_research_demo_inputs(tmp_path / "demo_inputs")

    weights = pd.read_csv(result.static_weights_path)

    assert set(weights.columns) == {"ticker", "weight"}
    assert abs(float(weights["weight"].sum()) - 1.0) < 1e-12


def test_demo_regime_policy_has_three_complete_regimes(tmp_path: Path) -> None:
    result = create_advanced_research_demo_inputs(tmp_path / "demo_inputs")

    policy = pd.read_csv(result.regime_policy_path)

    assert set(policy.columns) == {"regime", "ticker", "weight"}
    assert set(policy["regime"]) == {0, 1, 2}

    for _regime, regime_policy in policy.groupby("regime"):
        assert set(regime_policy["ticker"]) == {"SPY", "TLT", "GLD"}
        assert abs(float(regime_policy["weight"].sum()) - 1.0) < 1e-12


def test_demo_stress_periods_have_expected_columns(tmp_path: Path) -> None:
    result = create_advanced_research_demo_inputs(tmp_path / "demo_inputs")

    stress_periods = pd.read_csv(result.stress_periods_path)

    assert set(stress_periods.columns) == {"name", "start_date", "end_date"}
    assert len(stress_periods) == 3


def test_demo_factor_returns_have_expected_columns(tmp_path: Path) -> None:
    result = create_advanced_research_demo_inputs(tmp_path / "demo_inputs")

    factor_returns = pd.read_csv(result.factor_returns_path)

    assert set(factor_returns.columns) == {
        "date",
        "equity",
        "defensive",
        "real_asset",
    }
    assert len(factor_returns) == 180


def test_demo_inputs_can_run_advanced_export(tmp_path: Path) -> None:
    demo_inputs = create_advanced_research_demo_inputs(tmp_path / "demo_inputs")

    export_result = export_advanced_research_from_files(
        price_data_path=demo_inputs.price_data_path,
        static_weights_path=demo_inputs.static_weights_path,
        regime_policy_path=demo_inputs.regime_policy_path,
        stress_periods_path=demo_inputs.stress_periods_path,
        factor_returns_path=demo_inputs.factor_returns_path,
        output_dir=tmp_path / "advanced_export",
        n_regimes=3,
        feature_window=10,
        scenario_horizon=5,
        scenario_simulations=25,
        random_state=7,
        analyst="Jimena Chinchilla",
    )

    assert export_result.memo_path.exists()
    assert "regime_intelligence_profile" in export_result.exported_table_paths
    assert "stress_test_summary" in export_result.exported_table_paths
    assert "factor_exposure" in export_result.exported_table_paths
    assert "scenario_terminal_summary" in export_result.exported_table_paths


def test_create_advanced_research_demo_inputs_rejects_overwrite_false(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "demo_inputs"

    create_advanced_research_demo_inputs(output_dir)

    with pytest.raises(AdvancedResearchDemoInputError, match="overwrite=False"):
        create_advanced_research_demo_inputs(output_dir, overwrite=False)


def test_create_advanced_research_demo_inputs_rejects_file_output_path(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "not_a_directory.txt"
    output_path.write_text("already exists", encoding="utf-8")

    with pytest.raises(AdvancedResearchDemoInputError, match="not a directory"):
        create_advanced_research_demo_inputs(output_path)
