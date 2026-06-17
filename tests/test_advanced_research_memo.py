import pandas as pd
import pytest

from regime_risk_engine.research.advanced_memo import (
    AdvancedResearchMemoConfig,
    AdvancedResearchMemoError,
    AdvancedResearchMemoInputs,
    build_advanced_research_memo,
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
        transition_counts=pd.DataFrame(
            [[8, 2], [3, 7]],
            index=[0, 1],
            columns=[0, 1],
        ),
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
        regime_attribution=None,
        top_positive_asset="GLD",
        top_negative_asset="SPY",
        strongest_regime=None,
        weakest_regime=None,
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
        regime_exposure_table=None,
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


def test_build_advanced_research_memo() -> None:
    memo = build_advanced_research_memo(
        AdvancedResearchMemoInputs(
            base_memo="# Base Memo\n\nCore analysis.",
            regime_intelligence=make_regime_intelligence(),
            regime_transitions=make_regime_transitions(),
            stress_test=make_stress_test(),
            attribution=make_attribution(),
            factor_exposure=make_factor_exposure(),
            scenario_simulation=make_scenario_simulation(),
        )
    )

    assert "# Advanced Regime-Aware Portfolio Research Memo" in memo
    assert "## Base Research Memo" in memo
    assert "## Regime Intelligence" in memo
    assert "## Regime Transition Risk" in memo
    assert "## Stress-Period Analysis" in memo
    assert "## Strategy Attribution" in memo
    assert "## Factor Exposure Analysis" in memo
    assert "## Forward Regime Scenario Simulation" in memo
    assert "## Research Takeaway" in memo
    assert "## Advanced Research Limitations" in memo


def test_advanced_research_memo_custom_config() -> None:
    memo = build_advanced_research_memo(
        inputs=AdvancedResearchMemoInputs(
            base_memo="# Base Memo\n\nCore analysis.",
        ),
        config=AdvancedResearchMemoConfig(
            title="Investment Committee Memo",
            analyst="Jimena Chinchilla",
            include_limitations=False,
        ),
    )

    assert "# Investment Committee Memo" in memo
    assert "**Analyst:** Jimena Chinchilla" in memo
    assert "## Advanced Research Limitations" not in memo


def test_advanced_research_memo_accepts_base_only() -> None:
    memo = build_advanced_research_memo(
        AdvancedResearchMemoInputs(
            base_memo="# Base Memo\n\nCore analysis.",
        )
    )

    assert "Only the base research memo was supplied" in memo


def test_advanced_research_memo_rejects_empty_base_memo() -> None:
    with pytest.raises(AdvancedResearchMemoError, match="Base memo"):
        build_advanced_research_memo(
            AdvancedResearchMemoInputs(base_memo=" "),
        )


def test_advanced_research_memo_rejects_empty_title() -> None:
    with pytest.raises(AdvancedResearchMemoError, match="title"):
        build_advanced_research_memo(
            inputs=AdvancedResearchMemoInputs(
                base_memo="# Base Memo\n\nCore analysis.",
            ),
            config=AdvancedResearchMemoConfig(title=" "),
        )


def test_advanced_research_memo_rejects_empty_analyst() -> None:
    with pytest.raises(AdvancedResearchMemoError, match="Analyst"):
        build_advanced_research_memo(
            inputs=AdvancedResearchMemoInputs(
                base_memo="# Base Memo\n\nCore analysis.",
            ),
            config=AdvancedResearchMemoConfig(analyst=" "),
        )
