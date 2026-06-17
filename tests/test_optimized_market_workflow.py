import pandas as pd
import pytest

from regime_risk_engine.research.market_workflow import MarketResearchWorkflowConfig
from regime_risk_engine.research.optimized_workflow import (
    OptimizedMarketResearchWorkflowError,
    OptimizedMarketResearchWorkflowResult,
    run_optimized_market_research_workflow,
)
from regime_risk_engine.research.regime_optimizer import RegimePortfolioOptimizerConfig


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


def make_market_config() -> MarketResearchWorkflowConfig:
    return MarketResearchWorkflowConfig(
        n_regimes=3,
        feature_window=5,
        transaction_cost_bps=1.0,
        random_state=7,
    )


def make_optimizer_config() -> RegimePortfolioOptimizerConfig:
    return RegimePortfolioOptimizerConfig(
        max_weight=0.80,
        risk_aversion=0.25,
        cvar_penalty=0.25,
        turnover_penalty=0.05,
        n_candidate_portfolios=250,
        random_state=7,
    )


def test_run_optimized_market_research_workflow() -> None:
    result = run_optimized_market_research_workflow(
        price_data=make_price_data(),
        static_weights=make_static_weights(),
        market_config=make_market_config(),
        optimizer_config=make_optimizer_config(),
    )

    assert isinstance(result, OptimizedMarketResearchWorkflowResult)
    assert not result.initial_workflow_result.asset_returns.empty
    assert len(result.optimization_result.weight_table) == 3
    assert len(result.optimized_regime_policy) == 3
    assert not result.optimized_workflow_result.research_result.executive_summary == ""


def test_optimized_policy_weights_sum_to_one() -> None:
    result = run_optimized_market_research_workflow(
        price_data=make_price_data(),
        static_weights=make_static_weights(),
        market_config=make_market_config(),
        optimizer_config=make_optimizer_config(),
    )

    for regime_weights in result.optimized_regime_policy.values():
        assert abs(sum(regime_weights.values()) - 1.0) < 1e-8


def test_optimized_workflow_creates_dynamic_strategy_results() -> None:
    result = run_optimized_market_research_workflow(
        price_data=make_price_data(),
        static_weights=make_static_weights(),
        market_config=make_market_config(),
        optimizer_config=make_optimizer_config(),
    )

    return_comparison = result.optimized_workflow_result.strategy_comparison

    assert "static" in return_comparison.return_comparison.columns
    assert "dynamic" in return_comparison.return_comparison.columns
    assert not return_comparison.metric_summary.empty
    assert not return_comparison.metric_deltas.empty


def test_optimized_workflow_rejects_empty_price_data() -> None:
    with pytest.raises(OptimizedMarketResearchWorkflowError, match="Price data"):
        run_optimized_market_research_workflow(
            price_data=pd.DataFrame(),
            static_weights=make_static_weights(),
            market_config=make_market_config(),
            optimizer_config=make_optimizer_config(),
        )


def test_optimized_workflow_rejects_missing_ticker_column() -> None:
    price_data = make_price_data().drop(columns=["ticker"])

    with pytest.raises(OptimizedMarketResearchWorkflowError, match="ticker"):
        run_optimized_market_research_workflow(
            price_data=price_data,
            static_weights=make_static_weights(),
            market_config=make_market_config(),
            optimizer_config=make_optimizer_config(),
        )


def test_optimized_workflow_rejects_single_ticker() -> None:
    price_data = make_price_data()
    price_data = price_data[price_data["ticker"] == "SPY"]

    with pytest.raises(OptimizedMarketResearchWorkflowError, match="At least two"):
        run_optimized_market_research_workflow(
            price_data=price_data,
            static_weights={"SPY": 1.0},
            market_config=make_market_config(),
            optimizer_config=make_optimizer_config(),
        )
