from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.backtesting.engine import (
    BacktestResult,
    run_return_backtest,
)
from regime_risk_engine.risk.metrics import (
    DEFAULT_ANNUALIZATION_FACTOR,
    DEFAULT_CONFIDENCE_LEVEL,
    summarize_multiple_return_series,
)


class StrategyComparisonError(ValueError):
    """Raised when strategy comparisons cannot be calculated."""


@dataclass(frozen=True, slots=True)
class StrategyComparisonResult:
    """Container for static vs dynamic strategy comparison results."""

    static_backtest: BacktestResult
    dynamic_backtest: BacktestResult
    return_comparison: pd.DataFrame
    metric_summary: pd.DataFrame
    metric_deltas: pd.DataFrame


def compare_static_and_dynamic_backtests(
    returns: pd.DataFrame,
    static_weight_frame: pd.DataFrame,
    dynamic_weight_frame: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    risk_free_rate: float = 0.0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    return_col: str = "return",
    weight_lag: int = 1,
) -> StrategyComparisonResult:
    """Run and compare static and dynamic strategy backtests."""
    static_backtest = run_return_backtest(
        returns=returns,
        weight_frame=static_weight_frame,
        transaction_cost_bps=transaction_cost_bps,
        return_col=return_col,
        weight_lag=weight_lag,
    )
    dynamic_backtest = run_return_backtest(
        returns=returns,
        weight_frame=dynamic_weight_frame,
        transaction_cost_bps=transaction_cost_bps,
        return_col=return_col,
        weight_lag=weight_lag,
    )

    return_comparison = build_strategy_return_frame(
        {
            "static": static_backtest.net_returns,
            "dynamic": dynamic_backtest.net_returns,
        }
    )

    metric_summary = build_strategy_metric_summary(
        {
            "static": return_comparison["static"],
            "dynamic": return_comparison["dynamic"],
        },
        risk_free_rate=risk_free_rate,
        confidence_level=confidence_level,
        annualization_factor=annualization_factor,
    )

    metric_deltas = calculate_metric_deltas(
        metric_summary=metric_summary,
        benchmark_strategy="static",
        candidate_strategy="dynamic",
    )

    return StrategyComparisonResult(
        static_backtest=static_backtest,
        dynamic_backtest=dynamic_backtest,
        return_comparison=return_comparison,
        metric_summary=metric_summary,
        metric_deltas=metric_deltas,
    )


def build_strategy_return_frame(
    strategy_returns: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Align strategy return series on overlapping dates."""
    if not strategy_returns:
        raise StrategyComparisonError("At least one strategy return series is required")

    cleaned_returns: dict[str, pd.Series] = {}

    for strategy_name, returns in strategy_returns.items():
        clean_name = str(strategy_name).strip()

        if not clean_name:
            raise StrategyComparisonError("Strategy names must be non-empty")

        if clean_name in cleaned_returns:
            raise StrategyComparisonError(f"Duplicate strategy name: {clean_name}")

        cleaned_returns[clean_name] = _validate_strategy_returns(
            strategy_name=clean_name,
            returns=returns,
        )

    return_frame = pd.concat(cleaned_returns.values(), axis=1, join="inner")

    if return_frame.empty:
        raise StrategyComparisonError("Strategy returns have no overlapping dates")

    return_frame.index.name = "date"

    return return_frame.sort_index()


def build_strategy_metric_summary(
    strategy_returns: Mapping[str, pd.Series],
    risk_free_rate: float = 0.0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> pd.DataFrame:
    """Build risk metric summary for multiple strategy return series."""
    if not strategy_returns:
        raise StrategyComparisonError("At least one strategy return series is required")

    return_frame = build_strategy_return_frame(strategy_returns)

    aligned_returns = {
        strategy_name: return_frame[strategy_name]
        for strategy_name in return_frame.columns
    }

    summary = summarize_multiple_return_series(
        returns_by_name=aligned_returns,
        risk_free_rate=risk_free_rate,
        confidence_level=confidence_level,
        annualization_factor=annualization_factor,
    )
    summary.index.name = "strategy"

    return summary


def calculate_metric_deltas(
    metric_summary: pd.DataFrame,
    benchmark_strategy: str,
    candidate_strategy: str,
) -> pd.DataFrame:
    """Calculate candidate-minus-benchmark metric deltas."""
    _validate_metric_summary(metric_summary)

    benchmark = str(benchmark_strategy).strip()
    candidate = str(candidate_strategy).strip()

    if benchmark not in metric_summary.index:
        raise StrategyComparisonError(f"Benchmark strategy not found: {benchmark}")

    if candidate not in metric_summary.index:
        raise StrategyComparisonError(f"Candidate strategy not found: {candidate}")

    rows: list[dict[str, float | str]] = []

    for metric in metric_summary.columns:
        benchmark_value = _get_metric_value(
            metric_summary=metric_summary,
            strategy=benchmark,
            metric=str(metric),
        )
        candidate_value = _get_metric_value(
            metric_summary=metric_summary,
            strategy=candidate,
            metric=str(metric),
        )
        absolute_delta = candidate_value - benchmark_value

        if benchmark_value == 0:
            relative_delta = float("nan")
        else:
            relative_delta = absolute_delta / abs(benchmark_value)

        rows.append(
            {
                "metric": str(metric),
                "benchmark_strategy": benchmark,
                "candidate_strategy": candidate,
                "benchmark_value": benchmark_value,
                "candidate_value": candidate_value,
                "absolute_delta": absolute_delta,
                "relative_delta": relative_delta,
            }
        )

    return pd.DataFrame(rows)


def _validate_strategy_returns(
    strategy_name: str,
    returns: pd.Series,
) -> pd.Series:
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise StrategyComparisonError(
            f"Return series for {strategy_name} must use a DatetimeIndex"
        )

    if returns.empty:
        raise StrategyComparisonError(
            f"Return series for {strategy_name} cannot be empty"
        )

    if returns.index.has_duplicates:
        raise StrategyComparisonError(
            f"Return series for {strategy_name} contains duplicate dates"
        )

    clean_returns = pd.to_numeric(returns, errors="coerce")

    if clean_returns.isna().any():
        raise StrategyComparisonError(
            f"Return series for {strategy_name} contains missing values"
        )

    return pd.Series(
        clean_returns,
        index=returns.index,
        name=strategy_name,
    ).sort_index()


def _validate_metric_summary(metric_summary: pd.DataFrame) -> None:
    if metric_summary.empty:
        raise StrategyComparisonError("Metric summary cannot be empty")

    if metric_summary.index.has_duplicates:
        raise StrategyComparisonError("Metric summary contains duplicate strategies")

    if metric_summary.columns.has_duplicates:
        raise StrategyComparisonError("Metric summary contains duplicate metrics")


def _get_metric_value(
    metric_summary: pd.DataFrame,
    strategy: str,
    metric: str,
) -> float:
    try:
        return float(metric_summary.loc[strategy, metric])
    except (TypeError, ValueError) as error:
        raise StrategyComparisonError(
            f"Metric {metric} for strategy {strategy} is not numeric"
        ) from error
