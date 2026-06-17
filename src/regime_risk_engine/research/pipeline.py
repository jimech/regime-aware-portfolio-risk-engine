from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.backtesting.comparison import StrategyComparisonResult
from regime_risk_engine.backtesting.regime_evaluation import RegimeBacktestEvaluation
from regime_risk_engine.reporting.tables import (
    build_metric_delta_table,
    build_regime_metric_table,
    build_strategy_metric_table,
)
from regime_risk_engine.research.summary import (
    RegimeResearchSummary,
    StrategyResearchSummary,
    build_executive_research_summary,
    build_regime_research_summary,
    build_strategy_research_summary,
)


class InvestmentResearchPipelineError(ValueError):
    """Raised when the investment research pipeline cannot be run."""


@dataclass(frozen=True, slots=True)
class InvestmentResearchPipelineResult:
    """Container for investment research pipeline outputs."""

    strategy_metric_table: pd.DataFrame
    metric_delta_table: pd.DataFrame
    regime_metric_table: pd.DataFrame
    strategy_summary: StrategyResearchSummary
    regime_summary: RegimeResearchSummary
    executive_summary: str


def run_investment_research_pipeline(
    strategy_comparison: StrategyComparisonResult,
    regime_evaluation: RegimeBacktestEvaluation,
    benchmark_strategy: str = "static",
    candidate_strategy: str = "dynamic",
    primary_regime_metric: str = "sharpe_ratio",
) -> InvestmentResearchPipelineResult:
    """Run the investment research interpretation pipeline."""
    clean_benchmark = _validate_non_empty_string(
        benchmark_strategy,
        "Benchmark strategy",
    )
    clean_candidate = _validate_non_empty_string(
        candidate_strategy,
        "Candidate strategy",
    )
    clean_primary_metric = _validate_non_empty_string(
        primary_regime_metric,
        "Primary regime metric",
    )

    _validate_strategy_comparison(strategy_comparison)
    _validate_regime_evaluation(regime_evaluation)

    strategy_metric_table = build_strategy_metric_table(
        metric_summary=strategy_comparison.metric_summary,
    )
    metric_delta_table = build_metric_delta_table(
        metric_deltas=strategy_comparison.metric_deltas,
    )
    regime_metric_table = build_regime_metric_table(
        regime_metric_summary=regime_evaluation.metric_summary,
    )

    strategy_summary = build_strategy_research_summary(
        metric_delta_table=metric_delta_table,
    )
    regime_summary = build_regime_research_summary(
        regime_metric_table=regime_metric_table,
        candidate_strategy=clean_candidate,
        benchmark_strategy=clean_benchmark,
        primary_metric=clean_primary_metric,
    )
    executive_summary = build_executive_research_summary(
        strategy_summary=strategy_summary,
        regime_summary=regime_summary,
    )

    return InvestmentResearchPipelineResult(
        strategy_metric_table=strategy_metric_table,
        metric_delta_table=metric_delta_table,
        regime_metric_table=regime_metric_table,
        strategy_summary=strategy_summary,
        regime_summary=regime_summary,
        executive_summary=executive_summary,
    )


def _validate_strategy_comparison(
    strategy_comparison: StrategyComparisonResult,
) -> None:
    _validate_frame(
        strategy_comparison.return_comparison,
        "strategy return comparison",
    )
    _validate_frame(
        strategy_comparison.metric_summary,
        "strategy metric summary",
    )
    _validate_frame(
        strategy_comparison.metric_deltas,
        "strategy metric deltas",
    )


def _validate_regime_evaluation(
    regime_evaluation: RegimeBacktestEvaluation,
) -> None:
    _validate_frame(
        regime_evaluation.regime_return_frame,
        "regime return frame",
    )
    _validate_frame(
        regime_evaluation.metric_summary,
        "regime metric summary",
    )
    _validate_frame(
        regime_evaluation.metric_deltas,
        "regime metric deltas",
    )


def _validate_frame(frame: pd.DataFrame, frame_name: str) -> None:
    if frame.empty:
        raise InvestmentResearchPipelineError(f"{frame_name} cannot be empty")


def _validate_non_empty_string(value: str, label: str) -> str:
    clean_value = str(value).strip()

    if not clean_value:
        raise InvestmentResearchPipelineError(f"{label} must be non-empty")

    return clean_value
