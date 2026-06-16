import pandas as pd
import pytest

from regime_risk_engine.research.summary import (
    InvestmentResearchSummaryError,
    RegimeResearchSummary,
    StrategyResearchSummary,
    build_executive_research_summary,
    build_regime_research_summary,
    build_strategy_research_summary,
)


def make_metric_delta_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "cumulative_return",
                "sharpe_ratio",
                "annualized_volatility",
                "max_drawdown",
            ],
            "metric_label": [
                "Cumulative Return",
                "Sharpe Ratio",
                "Annualized Volatility",
                "Maximum Drawdown",
            ],
            "benchmark_strategy": [
                "static",
                "static",
                "static",
                "static",
            ],
            "candidate_strategy": [
                "dynamic",
                "dynamic",
                "dynamic",
                "dynamic",
            ],
            "benchmark_value": [
                0.08,
                0.90,
                0.18,
                -0.25,
            ],
            "candidate_value": [
                0.12,
                1.10,
                0.15,
                -0.18,
            ],
            "absolute_delta": [
                0.04,
                0.20,
                -0.03,
                0.07,
            ],
            "relative_delta": [
                0.50,
                0.2222,
                -0.1667,
                0.28,
            ],
        }
    )


def make_regime_metric_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regime": [
                0,
                0,
                1,
                1,
                2,
                2,
            ],
            "strategy": [
                "static",
                "dynamic",
                "static",
                "dynamic",
                "static",
                "dynamic",
            ],
            "observation_count": [
                100,
                100,
                80,
                80,
                60,
                60,
            ],
            "metric": [
                "sharpe_ratio",
                "sharpe_ratio",
                "sharpe_ratio",
                "sharpe_ratio",
                "sharpe_ratio",
                "sharpe_ratio",
            ],
            "metric_label": [
                "Sharpe Ratio",
                "Sharpe Ratio",
                "Sharpe Ratio",
                "Sharpe Ratio",
                "Sharpe Ratio",
                "Sharpe Ratio",
            ],
            "value": [
                1.0,
                1.2,
                -0.3,
                0.1,
                0.8,
                0.6,
            ],
        }
    )


def test_build_strategy_research_summary() -> None:
    summary = build_strategy_research_summary(make_metric_delta_table())

    assert isinstance(summary, StrategyResearchSummary)
    assert summary.benchmark_strategy == "static"
    assert summary.candidate_strategy == "dynamic"
    assert summary.total_metric_count == 4
    assert summary.favorable_metric_count == 3
    assert summary.unfavorable_metric_count == 1
    assert "broad improvement" in summary.overall_verdict
    assert len(summary.metric_assessments) == 4


def test_strategy_research_summary_marks_lower_risk_as_favorable() -> None:
    summary = build_strategy_research_summary(make_metric_delta_table())

    volatility_row = summary.metric_assessments[
        summary.metric_assessments["metric"] == "annualized_volatility"
    ].iloc[0]

    assert bool(volatility_row["is_favorable"]) is True


def test_strategy_research_summary_marks_worse_drawdown_as_unfavorable() -> None:
    summary = build_strategy_research_summary(make_metric_delta_table())

    drawdown_row = summary.metric_assessments[
        summary.metric_assessments["metric"] == "max_drawdown"
    ].iloc[0]

    assert bool(drawdown_row["is_favorable"]) is False


def test_build_regime_research_summary() -> None:
    summary = build_regime_research_summary(
        regime_metric_table=make_regime_metric_table(),
        candidate_strategy="dynamic",
        benchmark_strategy="static",
        primary_metric="sharpe_ratio",
    )

    assert isinstance(summary, RegimeResearchSummary)
    assert summary.best_regime == 1
    assert summary.worst_regime == 2
    assert "outperformed static in 2 of 3 regimes" in summary.conclusion
    assert len(summary.regime_summary) == 3


def test_build_executive_research_summary() -> None:
    strategy_summary = build_strategy_research_summary(make_metric_delta_table())
    regime_summary = build_regime_research_summary(
        regime_metric_table=make_regime_metric_table(),
        candidate_strategy="dynamic",
        benchmark_strategy="static",
    )

    executive_summary = build_executive_research_summary(
        strategy_summary=strategy_summary,
        regime_summary=regime_summary,
    )

    assert "dynamic strategy was compared against the static benchmark" in (
        executive_summary
    )
    assert "Regime-level analysis" in executive_summary


def test_strategy_summary_rejects_empty_delta_table() -> None:
    with pytest.raises(InvestmentResearchSummaryError, match="cannot be empty"):
        build_strategy_research_summary(pd.DataFrame())


def test_strategy_summary_rejects_missing_columns() -> None:
    metric_delta_table = make_metric_delta_table().drop(columns=["absolute_delta"])

    with pytest.raises(InvestmentResearchSummaryError, match="Missing metric delta"):
        build_strategy_research_summary(metric_delta_table)


def test_strategy_summary_rejects_multiple_candidate_strategies() -> None:
    metric_delta_table = make_metric_delta_table()
    metric_delta_table.loc[0, "candidate_strategy"] = "other"

    with pytest.raises(InvestmentResearchSummaryError, match="candidate strategy"):
        build_strategy_research_summary(metric_delta_table)


def test_regime_summary_rejects_empty_table() -> None:
    with pytest.raises(InvestmentResearchSummaryError, match="cannot be empty"):
        build_regime_research_summary(
            regime_metric_table=pd.DataFrame(),
            candidate_strategy="dynamic",
            benchmark_strategy="static",
        )


def test_regime_summary_rejects_missing_primary_metric() -> None:
    with pytest.raises(InvestmentResearchSummaryError, match="Primary regime metric"):
        build_regime_research_summary(
            regime_metric_table=make_regime_metric_table(),
            candidate_strategy="dynamic",
            benchmark_strategy="static",
            primary_metric="sortino_ratio",
        )


def test_regime_summary_rejects_missing_candidate_strategy() -> None:
    with pytest.raises(InvestmentResearchSummaryError, match="Candidate strategy"):
        build_regime_research_summary(
            regime_metric_table=make_regime_metric_table(),
            candidate_strategy="missing",
            benchmark_strategy="static",
        )
