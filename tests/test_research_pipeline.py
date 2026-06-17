import pandas as pd
import pytest

from regime_risk_engine.backtesting.comparison import StrategyComparisonResult
from regime_risk_engine.backtesting.engine import BacktestResult
from regime_risk_engine.backtesting.regime_evaluation import RegimeBacktestEvaluation
from regime_risk_engine.research.pipeline import (
    InvestmentResearchPipelineError,
    InvestmentResearchPipelineResult,
    run_investment_research_pipeline,
)


def make_dates() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=4, freq="D")


def make_backtest_result(name: str) -> BacktestResult:
    dates = make_dates()

    gross_returns = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=dates,
        name=name,
    )
    transaction_costs = pd.Series(
        [0.0, 0.0001, 0.0002, 0.0001],
        index=dates,
        name="transaction_cost",
    )
    turnover = pd.Series(
        [0.0, 0.1, 0.2, 0.1],
        index=dates,
        name="turnover",
    )
    weights = pd.DataFrame(
        {
            "SPY": [0.6, 0.6, 0.5, 0.5],
            "TLT": [0.4, 0.4, 0.5, 0.5],
        },
        index=dates,
    )

    return BacktestResult(
        gross_returns=gross_returns,
        net_returns=gross_returns - transaction_costs,
        turnover=turnover,
        transaction_costs=transaction_costs,
        applied_weights=weights,
    )


def make_strategy_comparison() -> StrategyComparisonResult:
    dates = make_dates()

    return_comparison = pd.DataFrame(
        {
            "static": [0.01, 0.02, -0.01, 0.03],
            "dynamic": [0.015, 0.018, 0.00, 0.035],
        },
        index=dates,
    )

    metric_summary = pd.DataFrame(
        {
            "cumulative_return": [0.05, 0.07],
            "sharpe_ratio": [1.0, 1.2],
            "annualized_volatility": [0.18, 0.15],
            "max_drawdown": [-0.10, -0.08],
        },
        index=["static", "dynamic"],
    )

    metric_deltas = pd.DataFrame(
        {
            "metric": [
                "cumulative_return",
                "sharpe_ratio",
                "annualized_volatility",
                "max_drawdown",
            ],
            "benchmark_strategy": ["static", "static", "static", "static"],
            "candidate_strategy": ["dynamic", "dynamic", "dynamic", "dynamic"],
            "benchmark_value": [0.05, 1.0, 0.18, -0.10],
            "candidate_value": [0.07, 1.2, 0.15, -0.08],
            "absolute_delta": [0.02, 0.2, -0.03, 0.02],
            "relative_delta": [0.40, 0.20, -0.1667, 0.20],
        }
    )

    return StrategyComparisonResult(
        static_backtest=make_backtest_result("static"),
        dynamic_backtest=make_backtest_result("dynamic"),
        return_comparison=return_comparison,
        metric_summary=metric_summary,
        metric_deltas=metric_deltas,
    )


def make_regime_evaluation() -> RegimeBacktestEvaluation:
    dates = make_dates()

    regime_return_frame = pd.DataFrame(
        {
            "static": [0.01, 0.02, -0.01, 0.03],
            "dynamic": [0.015, 0.018, 0.00, 0.035],
            "regime": [0, 0, 1, 1],
        },
        index=dates,
    )

    metric_summary = pd.DataFrame(
        {
            "regime": [0, 0, 1, 1],
            "strategy": ["static", "dynamic", "static", "dynamic"],
            "observation_count": [2, 2, 2, 2],
            "cumulative_return": [0.03, 0.034, 0.02, 0.035],
            "sharpe_ratio": [0.8, 1.0, 0.4, 0.7],
        }
    )

    metric_deltas = pd.DataFrame(
        {
            "regime": [0, 1],
            "metric": ["sharpe_ratio", "sharpe_ratio"],
            "benchmark_strategy": ["static", "static"],
            "candidate_strategy": ["dynamic", "dynamic"],
            "benchmark_value": [0.8, 0.4],
            "candidate_value": [1.0, 0.7],
            "absolute_delta": [0.2, 0.3],
            "relative_delta": [0.25, 0.75],
        }
    )

    return RegimeBacktestEvaluation(
        regime_return_frame=regime_return_frame,
        metric_summary=metric_summary,
        metric_deltas=metric_deltas,
    )


def test_run_investment_research_pipeline() -> None:
    result = run_investment_research_pipeline(
        strategy_comparison=make_strategy_comparison(),
        regime_evaluation=make_regime_evaluation(),
        benchmark_strategy="static",
        candidate_strategy="dynamic",
        primary_regime_metric="sharpe_ratio",
    )

    assert isinstance(result, InvestmentResearchPipelineResult)
    assert not result.strategy_metric_table.empty
    assert not result.metric_delta_table.empty
    assert not result.regime_metric_table.empty
    assert result.strategy_summary.candidate_strategy == "dynamic"
    assert result.strategy_summary.benchmark_strategy == "static"
    assert result.strategy_summary.favorable_metric_count == 4
    assert result.regime_summary.best_regime == 1
    assert "dynamic strategy was compared against the static benchmark" in (
        result.executive_summary
    )


def test_research_pipeline_rejects_empty_benchmark_strategy() -> None:
    with pytest.raises(InvestmentResearchPipelineError, match="Benchmark strategy"):
        run_investment_research_pipeline(
            strategy_comparison=make_strategy_comparison(),
            regime_evaluation=make_regime_evaluation(),
            benchmark_strategy=" ",
            candidate_strategy="dynamic",
        )


def test_research_pipeline_rejects_empty_candidate_strategy() -> None:
    with pytest.raises(InvestmentResearchPipelineError, match="Candidate strategy"):
        run_investment_research_pipeline(
            strategy_comparison=make_strategy_comparison(),
            regime_evaluation=make_regime_evaluation(),
            benchmark_strategy="static",
            candidate_strategy=" ",
        )


def test_research_pipeline_rejects_empty_primary_metric() -> None:
    with pytest.raises(InvestmentResearchPipelineError, match="Primary regime metric"):
        run_investment_research_pipeline(
            strategy_comparison=make_strategy_comparison(),
            regime_evaluation=make_regime_evaluation(),
            benchmark_strategy="static",
            candidate_strategy="dynamic",
            primary_regime_metric=" ",
        )


def test_research_pipeline_rejects_empty_strategy_metric_summary() -> None:
    strategy_comparison = make_strategy_comparison()
    broken_comparison = StrategyComparisonResult(
        static_backtest=strategy_comparison.static_backtest,
        dynamic_backtest=strategy_comparison.dynamic_backtest,
        return_comparison=strategy_comparison.return_comparison,
        metric_summary=pd.DataFrame(),
        metric_deltas=strategy_comparison.metric_deltas,
    )

    with pytest.raises(InvestmentResearchPipelineError, match="metric summary"):
        run_investment_research_pipeline(
            strategy_comparison=broken_comparison,
            regime_evaluation=make_regime_evaluation(),
        )


def test_research_pipeline_rejects_empty_regime_metric_summary() -> None:
    regime_evaluation = make_regime_evaluation()
    broken_evaluation = RegimeBacktestEvaluation(
        regime_return_frame=regime_evaluation.regime_return_frame,
        metric_summary=pd.DataFrame(),
        metric_deltas=regime_evaluation.metric_deltas,
    )

    with pytest.raises(InvestmentResearchPipelineError, match="regime metric summary"):
        run_investment_research_pipeline(
            strategy_comparison=make_strategy_comparison(),
            regime_evaluation=broken_evaluation,
        )
