from dataclasses import dataclass
from math import sqrt

import pandas as pd


class StressTestingError(ValueError):
    """Raised when stress-period analysis cannot be completed."""


@dataclass(frozen=True, slots=True)
class StressPeriod:
    """Named historical or synthetic stress period."""

    name: str
    start_date: str
    end_date: str


@dataclass(frozen=True, slots=True)
class StressTestSummary:
    """Stress-period strategy analysis result."""

    summary_table: pd.DataFrame
    best_period: str | None
    worst_period: str | None
    protected_capital_period_count: int
    total_period_count: int
    narrative: str


def build_stress_test_summary(
    return_frame: pd.DataFrame,
    stress_periods: list[StressPeriod],
    benchmark_strategy: str = "static",
    candidate_strategy: str = "dynamic",
    annualization_factor: int = 252,
) -> StressTestSummary:
    """Compare candidate and benchmark strategy behavior during stress periods."""
    clean_benchmark = _validate_non_empty_string(
        benchmark_strategy,
        "Benchmark strategy",
    )
    clean_candidate = _validate_non_empty_string(
        candidate_strategy,
        "Candidate strategy",
    )

    _validate_inputs(
        return_frame=return_frame,
        stress_periods=stress_periods,
        benchmark_strategy=clean_benchmark,
        candidate_strategy=clean_candidate,
        annualization_factor=annualization_factor,
    )

    rows = []

    for period in stress_periods:
        rows.append(
            _analyze_stress_period(
                return_frame=return_frame,
                period=period,
                benchmark_strategy=clean_benchmark,
                candidate_strategy=clean_candidate,
                annualization_factor=annualization_factor,
            )
        )

    summary_table = pd.DataFrame(rows)

    best_period = _extract_period_by_sort(
        summary_table=summary_table,
        sort_column="return_delta",
        ascending=False,
    )
    worst_period = _extract_period_by_sort(
        summary_table=summary_table,
        sort_column="return_delta",
        ascending=True,
    )

    protected_capital_period_count = int(summary_table["protected_capital"].sum())
    total_period_count = len(summary_table)

    narrative = _build_narrative(
        summary_table=summary_table,
        benchmark_strategy=clean_benchmark,
        candidate_strategy=clean_candidate,
        best_period=best_period,
        worst_period=worst_period,
        protected_capital_period_count=protected_capital_period_count,
        total_period_count=total_period_count,
    )

    return StressTestSummary(
        summary_table=summary_table,
        best_period=best_period,
        worst_period=worst_period,
        protected_capital_period_count=protected_capital_period_count,
        total_period_count=total_period_count,
        narrative=narrative,
    )


def _validate_inputs(
    return_frame: pd.DataFrame,
    stress_periods: list[StressPeriod],
    benchmark_strategy: str,
    candidate_strategy: str,
    annualization_factor: int,
) -> None:
    if return_frame.empty:
        raise StressTestingError("Return frame cannot be empty")

    if not isinstance(return_frame.index, pd.DatetimeIndex):
        raise StressTestingError("Return frame index must be a DatetimeIndex")

    if not stress_periods:
        raise StressTestingError("At least one stress period is required")

    missing_columns = {
        benchmark_strategy,
        candidate_strategy,
    }.difference(return_frame.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise StressTestingError(f"Missing strategy return column(s): {missing}")

    strategy_returns = return_frame[[benchmark_strategy, candidate_strategy]].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if strategy_returns.isna().any().any():
        raise StressTestingError("Strategy returns must be numeric and complete")

    if annualization_factor <= 0:
        raise StressTestingError("Annualization factor must be positive")

    for period in stress_periods:
        _validate_period(period)


def _validate_period(period: StressPeriod) -> None:
    if not period.name.strip():
        raise StressTestingError("Stress period name must be non-empty")

    start_date = pd.Timestamp(period.start_date)
    end_date = pd.Timestamp(period.end_date)

    if start_date > end_date:
        raise StressTestingError(
            f"Stress period start date cannot be after end date: {period.name}"
        )


def _analyze_stress_period(
    return_frame: pd.DataFrame,
    period: StressPeriod,
    benchmark_strategy: str,
    candidate_strategy: str,
    annualization_factor: int,
) -> dict[str, object]:
    period_frame = _slice_period(
        return_frame=return_frame,
        period=period,
    )

    benchmark_metrics = _calculate_return_metrics(
        returns=period_frame[benchmark_strategy],
        annualization_factor=annualization_factor,
    )
    candidate_metrics = _calculate_return_metrics(
        returns=period_frame[candidate_strategy],
        annualization_factor=annualization_factor,
    )

    return_delta = (
        candidate_metrics["cumulative_return"] - benchmark_metrics["cumulative_return"]
    )
    drawdown_delta = (
        candidate_metrics["max_drawdown"] - benchmark_metrics["max_drawdown"]
    )
    volatility_delta = (
        candidate_metrics["annualized_volatility"]
        - benchmark_metrics["annualized_volatility"]
    )

    dominant_regime = _calculate_dominant_regime(period_frame)

    return {
        "period_name": period.name,
        "start_date": period_frame.index.min().date().isoformat(),
        "end_date": period_frame.index.max().date().isoformat(),
        "observation_count": int(len(period_frame)),
        "dominant_regime": dominant_regime,
        "benchmark_cumulative_return": benchmark_metrics["cumulative_return"],
        "candidate_cumulative_return": candidate_metrics["cumulative_return"],
        "return_delta": return_delta,
        "benchmark_max_drawdown": benchmark_metrics["max_drawdown"],
        "candidate_max_drawdown": candidate_metrics["max_drawdown"],
        "drawdown_delta": drawdown_delta,
        "benchmark_volatility": benchmark_metrics["annualized_volatility"],
        "candidate_volatility": candidate_metrics["annualized_volatility"],
        "volatility_delta": volatility_delta,
        "benchmark_sharpe": benchmark_metrics["sharpe_ratio"],
        "candidate_sharpe": candidate_metrics["sharpe_ratio"],
        "sharpe_delta": (
            candidate_metrics["sharpe_ratio"] - benchmark_metrics["sharpe_ratio"]
        ),
        "protected_capital": bool(return_delta > 0.0 and drawdown_delta >= 0.0),
        "assessment": _build_period_assessment(
            return_delta=return_delta,
            drawdown_delta=drawdown_delta,
            volatility_delta=volatility_delta,
        ),
    }


def _slice_period(
    return_frame: pd.DataFrame,
    period: StressPeriod,
) -> pd.DataFrame:
    start_date = pd.Timestamp(period.start_date)
    end_date = pd.Timestamp(period.end_date)

    period_frame = return_frame.loc[
        (return_frame.index >= start_date) & (return_frame.index <= end_date)
    ].copy()

    if period_frame.empty:
        raise StressTestingError(
            f"Stress period has no overlapping observations: {period.name}"
        )

    return period_frame


def _calculate_return_metrics(
    returns: pd.Series,
    annualization_factor: int,
) -> dict[str, float]:
    clean_returns = returns.dropna().astype(float)

    if clean_returns.empty:
        raise StressTestingError("Cannot calculate stress metrics on empty returns")

    cumulative_return = float((1.0 + clean_returns).prod() - 1.0)

    if len(clean_returns) < 2:
        annualized_volatility = 0.0
    else:
        annualized_volatility = float(
            clean_returns.std(ddof=1) * sqrt(annualization_factor)
        )

    if annualized_volatility == 0.0:
        sharpe_ratio = 0.0
    else:
        annualized_return = float(
            (1.0 + cumulative_return) ** (annualization_factor / len(clean_returns))
            - 1.0
        )
        sharpe_ratio = float(annualized_return / annualized_volatility)

    wealth = (1.0 + clean_returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "cumulative_return": cumulative_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def _calculate_dominant_regime(period_frame: pd.DataFrame) -> int | None:
    if "regime" not in period_frame.columns:
        return None

    regime_counts = period_frame["regime"].dropna().astype(int).value_counts()

    if regime_counts.empty:
        return None

    return int(regime_counts.index[0])


def _build_period_assessment(
    return_delta: float,
    drawdown_delta: float,
    volatility_delta: float,
) -> str:
    if return_delta > 0.0 and drawdown_delta >= 0.0:
        return "Protected capital"

    if return_delta > 0.0 and volatility_delta <= 0.0:
        return "Improved return with lower volatility"

    if return_delta > 0.0:
        return "Improved return with higher risk"

    if drawdown_delta >= 0.0:
        return "Reduced drawdown but lagged return"

    return "Underperformed during stress"


def _extract_period_by_sort(
    summary_table: pd.DataFrame,
    sort_column: str,
    ascending: bool,
) -> str | None:
    if summary_table.empty:
        return None

    sorted_table = summary_table.sort_values(sort_column, ascending=ascending)

    return str(sorted_table.iloc[0]["period_name"])


def _build_narrative(
    summary_table: pd.DataFrame,
    benchmark_strategy: str,
    candidate_strategy: str,
    best_period: str | None,
    worst_period: str | None,
    protected_capital_period_count: int,
    total_period_count: int,
) -> str:
    average_return_delta = float(summary_table["return_delta"].mean())
    average_drawdown_delta = float(summary_table["drawdown_delta"].mean())

    narrative = (
        f"Stress-period analysis compared {candidate_strategy} against "
        f"{benchmark_strategy} across {total_period_count} stress windows. "
        f"The candidate strategy protected capital in "
        f"{protected_capital_period_count} of {total_period_count} periods. "
        f"Average excess return across stress periods was "
        f"{average_return_delta:.4f}, and average drawdown improvement was "
        f"{average_drawdown_delta:.4f}."
    )

    if best_period is not None:
        narrative += f" The strongest stress-period result was {best_period}."

    if worst_period is not None:
        narrative += f" The weakest stress-period result was {worst_period}."

    return narrative


def _validate_non_empty_string(value: str, label: str) -> str:
    clean_value = str(value).strip()

    if not clean_value:
        raise StressTestingError(f"{label} must be non-empty")

    return clean_value
