import pandas as pd
import pytest

from regime_risk_engine.research.advanced_memo import AdvancedResearchMemoConfig
from regime_risk_engine.research.advanced_workflow import (
    AdvancedResearchWorkflowError,
    AdvancedResearchWorkflowResult,
    build_advanced_research_workflow,
)
from regime_risk_engine.research.market_workflow import (
    MarketResearchWorkflowConfig,
    run_market_research_workflow,
)
from regime_risk_engine.research.memo import MarketResearchMemoConfig
from regime_risk_engine.research.scenario_simulation import (
    RegimeScenarioSimulationConfig,
)
from regime_risk_engine.research.stress_testing import StressPeriod


def make_price_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rows = []

    spy_price = 100.0
    tlt_price = 100.0
    gld_price = 100.0

    for index, date in enumerate(dates):
        if index < 20:
            spy_return = 0.004
            tlt_return = 0.0005
            gld_return = 0.0002
        elif index < 40:
            spy_return = -0.006
            tlt_return = 0.003
            gld_return = 0.001
        else:
            spy_return = 0.0005
            tlt_return = -0.001
            gld_return = 0.004

        spy_price *= 1.0 + spy_return
        tlt_price *= 1.0 + tlt_return
        gld_price *= 1.0 + gld_return

        rows.extend(
            [
                {"date": date, "ticker": "SPY", "adjusted_close": spy_price},
                {"date": date, "ticker": "TLT", "adjusted_close": tlt_price},
                {"date": date, "ticker": "GLD", "adjusted_close": gld_price},
            ]
        )

    return pd.DataFrame(rows)


def make_static_weights() -> dict[str, float]:
    return {
        "SPY": 0.60,
        "TLT": 0.30,
        "GLD": 0.10,
    }


def make_regime_policy() -> dict[int, dict[str, float]]:
    return {
        0: {"SPY": 0.70, "TLT": 0.20, "GLD": 0.10},
        1: {"SPY": 0.30, "TLT": 0.50, "GLD": 0.20},
        2: {"SPY": 0.40, "TLT": 0.20, "GLD": 0.40},
    }


def make_market_result():
    return run_market_research_workflow(
        price_data=make_price_data(),
        static_weights=make_static_weights(),
        regime_weight_policy=make_regime_policy(),
        config=MarketResearchWorkflowConfig(
            n_regimes=3,
            feature_window=5,
            transaction_cost_bps=1.0,
            random_state=7,
        ),
    )


def make_stress_periods(market_result) -> list[StressPeriod]:
    dates = market_result.asset_returns.index

    return [
        StressPeriod(
            name="Full sample stress review",
            start_date=dates.min().date().isoformat(),
            end_date=dates.max().date().isoformat(),
        )
    ]


def make_factor_returns(market_result) -> pd.DataFrame:
    asset_returns = market_result.asset_returns

    return pd.DataFrame(
        {
            "equity": asset_returns["SPY"],
            "defensive": asset_returns["TLT"],
            "real_asset": asset_returns["GLD"],
        },
        index=asset_returns.index,
    )


def test_build_advanced_research_workflow() -> None:
    market_result = make_market_result()

    result = build_advanced_research_workflow(
        market_result=market_result,
        stress_periods=make_stress_periods(market_result),
        factor_returns=make_factor_returns(market_result),
        market_memo_config=MarketResearchMemoConfig(
            title="Base Market Memo",
            analyst="Jimena Chinchilla",
        ),
        advanced_memo_config=AdvancedResearchMemoConfig(
            title="Advanced IC Memo",
            analyst="Jimena Chinchilla",
        ),
        scenario_config=RegimeScenarioSimulationConfig(
            horizon=5,
            n_simulations=25,
            random_state=7,
        ),
    )

    assert isinstance(result, AdvancedResearchWorkflowResult)
    assert result.base_memo
    assert not result.regime_intelligence.profile_table.empty
    assert not result.regime_transitions.transition_probabilities.empty
    assert result.stress_test is not None
    assert not result.attribution.asset_attribution.empty
    assert result.factor_exposure is not None
    assert not result.scenario_simulation.terminal_summary.empty
    assert "# Advanced IC Memo" in result.advanced_memo
    assert "## Regime Intelligence" in result.advanced_memo
    assert "## Stress-Period Analysis" in result.advanced_memo
    assert "## Factor Exposure Analysis" in result.advanced_memo


def test_advanced_research_workflow_works_without_optional_inputs() -> None:
    result = build_advanced_research_workflow(
        market_result=make_market_result(),
        scenario_config=RegimeScenarioSimulationConfig(
            horizon=5,
            n_simulations=25,
            random_state=7,
        ),
    )

    assert result.stress_test is None
    assert result.factor_exposure is None
    assert "## Regime Intelligence" in result.advanced_memo
    assert "## Strategy Attribution" in result.advanced_memo
    assert "## Forward Regime Scenario Simulation" in result.advanced_memo


def test_advanced_research_workflow_rejects_empty_asset_returns() -> None:
    market_result = make_market_result()
    broken_result = type(market_result)(
        asset_returns=pd.DataFrame(),
        feature_matrix=market_result.feature_matrix,
        regime_labels=market_result.regime_labels,
        static_weight_frame=market_result.static_weight_frame,
        dynamic_target_weight_frame=market_result.dynamic_target_weight_frame,
        dynamic_applied_weight_frame=market_result.dynamic_applied_weight_frame,
        strategy_comparison=market_result.strategy_comparison,
        regime_evaluation=market_result.regime_evaluation,
        research_result=market_result.research_result,
    )

    with pytest.raises(AdvancedResearchWorkflowError, match="Asset returns"):
        build_advanced_research_workflow(broken_result)


def test_advanced_research_workflow_rejects_missing_strategy_columns() -> None:
    market_result = make_market_result()
    comparison = market_result.strategy_comparison
    broken_comparison = type(comparison)(
        static_backtest=comparison.static_backtest,
        dynamic_backtest=comparison.dynamic_backtest,
        return_comparison=comparison.return_comparison.rename(
            columns={"dynamic": "candidate"}
        ),
        metric_summary=comparison.metric_summary,
        metric_deltas=comparison.metric_deltas,
    )
    broken_result = type(market_result)(
        asset_returns=market_result.asset_returns,
        feature_matrix=market_result.feature_matrix,
        regime_labels=market_result.regime_labels,
        static_weight_frame=market_result.static_weight_frame,
        dynamic_target_weight_frame=market_result.dynamic_target_weight_frame,
        dynamic_applied_weight_frame=market_result.dynamic_applied_weight_frame,
        strategy_comparison=broken_comparison,
        regime_evaluation=market_result.regime_evaluation,
        research_result=market_result.research_result,
    )

    with pytest.raises(AdvancedResearchWorkflowError, match="Missing strategy"):
        build_advanced_research_workflow(broken_result)
