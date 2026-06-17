import pandas as pd
import pytest

from regime_risk_engine.research.advanced_export import (
    AdvancedResearchExportError,
    AdvancedResearchExportResult,
    export_advanced_research_package,
)
from regime_risk_engine.research.advanced_memo import (
    AdvancedResearchMemoConfig,
    AdvancedResearchMemoInputs,
)
from regime_risk_engine.research.attribution import StrategyAttributionSummary
from regime_risk_engine.research.factor_exposure import FactorExposureSummary
from regime_risk_engine.research.regime_intelligence import RegimeIntelligenceSummary
from regime_risk_engine.research.regime_transitions import RegimeTransitionSummary
from regime_risk_engine.research.scenario_simulation import (
    RegimeScenarioSimulationResult,
)
from regime_risk_engine.research.stress_testing import StressTestSummary


def make_regime_intelligence() -> RegimeIntelligenceSummary:
    return RegimeIntelligenceSummary(
        profile_table=pd.DataFrame(
            {
                "regime": [0, 1],
                "label": ["Growth / risk-on", "Defensive / stress"],
                "annualized_return": [0.12, -0.08],
                "annualized_volatility": [0.10, 0.25],
                "max_drawdown": [-0.03, -0.18],
                "best_asset": ["SPY", "TLT"],
                "worst_asset": ["TLT", "SPY"],
            }
        ),
        strongest_regime=0,
        weakest_regime=1,
        stress_regime=1,
        narrative="Regime intelligence narrative.",
    )


def make_regime_transitions() -> RegimeTransitionSummary:
    return RegimeTransitionSummary(
        transition_counts=pd.DataFrame([[8, 2], [3, 7]], index=[0, 1], columns=[0, 1]),
        transition_probabilities=pd.DataFrame(
            [[0.8, 0.2], [0.3, 0.7]],
            index=[0, 1],
            columns=[0, 1],
        ),
        regime_persistence=pd.DataFrame(
            {
                "regime": [0, 1],
                "persistence_probability": [0.8, 0.7],
                "expected_duration": [5.0, 3.3333],
                "transition_observation_count": [10, 10],
            }
        ),
        regime_durations=pd.DataFrame(
            {
                "regime": [0, 1],
                "duration": [5, 4],
            }
        ),
        most_likely_next_regime=pd.DataFrame(
            {
                "current_regime": [0, 1],
                "most_likely_next_regime": [0, 1],
                "transition_probability": [0.8, 0.7],
            }
        ),
        most_persistent_regime=0,
        least_persistent_regime=1,
        narrative="Regime transition narrative.",
    )


def make_stress_test() -> StressTestSummary:
    return StressTestSummary(
        summary_table=pd.DataFrame(
            {
                "period_name": ["Stress A"],
                "return_delta": [0.03],
                "drawdown_delta": [0.04],
                "volatility_delta": [-0.02],
                "assessment": ["Protected capital"],
            }
        ),
        best_period="Stress A",
        worst_period="Stress A",
        protected_capital_period_count=1,
        total_period_count=1,
        narrative="Stress test narrative.",
    )


def make_attribution() -> StrategyAttributionSummary:
    return StrategyAttributionSummary(
        asset_attribution=pd.DataFrame(
            {
                "asset": ["GLD", "SPY"],
                "average_active_weight": [0.20, -0.20],
                "active_return_contribution": [0.04, -0.01],
            }
        ),
        regime_attribution=pd.DataFrame(
            {
                "regime": [0, 1],
                "active_return_contribution": [0.03, 0.01],
            }
        ),
        top_positive_asset="GLD",
        top_negative_asset="SPY",
        strongest_regime=0,
        weakest_regime=1,
        narrative="Attribution narrative.",
    )


def make_factor_exposure() -> FactorExposureSummary:
    return FactorExposureSummary(
        exposure_table=pd.DataFrame(
            {
                "strategy": ["static", "dynamic"],
                "equity_beta": [0.7, 0.4],
            }
        ),
        regime_exposure_table=pd.DataFrame(
            {
                "regime": [0, 1],
                "strategy": ["dynamic", "dynamic"],
                "equity_beta": [0.6, 0.2],
            }
        ),
        dominant_factor_by_strategy=pd.DataFrame(
            {
                "strategy": ["static", "dynamic"],
                "dominant_factor": ["equity", "defensive"],
                "dominant_beta": [0.7, 0.5],
            }
        ),
        narrative="Factor exposure narrative.",
    )


def make_scenario_simulation() -> RegimeScenarioSimulationResult:
    return RegimeScenarioSimulationResult(
        simulated_paths=pd.DataFrame(
            {
                "simulation_id": [0, 0],
                "step": [1, 2],
                "regime": [0, 1],
                "static_wealth": [1.01, 1.00],
                "dynamic_wealth": [1.02, 1.03],
            }
        ),
        terminal_summary=pd.DataFrame(
            {
                "strategy": ["static", "dynamic"],
                "mean_terminal_return": [0.01, 0.03],
                "probability_of_loss": [0.40, 0.20],
                "var_95": [-0.05, -0.03],
                "cvar_95": [-0.08, -0.04],
            }
        ),
        transition_probabilities=pd.DataFrame(
            [[0.8, 0.2], [0.3, 0.7]],
            index=[0, 1],
            columns=[0, 1],
        ),
        regime_usage=pd.DataFrame(
            {
                "regime": [0, 1],
                "simulated_frequency": [0.55, 0.45],
            }
        ),
        narrative="Scenario simulation narrative.",
    )


def make_inputs() -> AdvancedResearchMemoInputs:
    return AdvancedResearchMemoInputs(
        base_memo="# Base Memo\n\nCore analysis.",
        regime_intelligence=make_regime_intelligence(),
        regime_transitions=make_regime_transitions(),
        stress_test=make_stress_test(),
        attribution=make_attribution(),
        factor_exposure=make_factor_exposure(),
        scenario_simulation=make_scenario_simulation(),
    )


def test_export_advanced_research_package(tmp_path) -> None:
    export_result = export_advanced_research_package(
        inputs=make_inputs(),
        output_dir=tmp_path / "advanced_export",
    )

    assert isinstance(export_result, AdvancedResearchExportResult)
    assert export_result.output_dir.exists()
    assert export_result.memo_path.exists()
    assert export_result.exported_table_paths

    memo = export_result.memo_path.read_text(encoding="utf-8")

    assert "# Advanced Regime-Aware Portfolio Research Memo" in memo
    assert "## Regime Intelligence" in memo
    assert "regime_intelligence_profile" in export_result.exported_table_paths
    assert "scenario_terminal_summary" in export_result.exported_table_paths

    for path in export_result.exported_table_paths.values():
        assert path.exists()


def test_export_advanced_research_package_with_custom_config(tmp_path) -> None:
    export_result = export_advanced_research_package(
        inputs=make_inputs(),
        output_dir=tmp_path / "custom_export",
        config=AdvancedResearchMemoConfig(
            title="Investment Committee Package",
            analyst="Jimena Chinchilla",
            include_limitations=False,
        ),
    )

    memo = export_result.memo_path.read_text(encoding="utf-8")

    assert "# Investment Committee Package" in memo
    assert "**Analyst:** Jimena Chinchilla" in memo
    assert "## Advanced Research Limitations" not in memo


def test_export_advanced_research_package_base_only(tmp_path) -> None:
    export_result = export_advanced_research_package(
        inputs=AdvancedResearchMemoInputs(
            base_memo="# Base Memo\n\nCore analysis.",
        ),
        output_dir=tmp_path / "base_only",
    )

    assert export_result.memo_path.exists()
    assert export_result.exported_table_paths == {}


def test_export_advanced_research_package_rejects_existing_files_without_overwrite(
    tmp_path,
) -> None:
    output_dir = tmp_path / "existing_export"

    export_advanced_research_package(
        inputs=make_inputs(),
        output_dir=output_dir,
    )

    with pytest.raises(AdvancedResearchExportError, match="overwrite=False"):
        export_advanced_research_package(
            inputs=make_inputs(),
            output_dir=output_dir,
            overwrite=False,
        )


def test_export_advanced_research_package_rejects_file_output_path(tmp_path) -> None:
    output_path = tmp_path / "not_a_directory.txt"
    output_path.write_text("already exists", encoding="utf-8")

    with pytest.raises(AdvancedResearchExportError, match="not a directory"):
        export_advanced_research_package(
            inputs=make_inputs(),
            output_dir=output_path,
        )
