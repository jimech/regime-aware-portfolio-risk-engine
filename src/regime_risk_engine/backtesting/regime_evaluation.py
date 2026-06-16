from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.risk.metrics import (
    DEFAULT_ANNUALIZATION_FACTOR,
    DEFAULT_CONFIDENCE_LEVEL,
    summarize_risk_metrics,
)


class RegimeBacktestEvaluationError(ValueError):
    """Raised when regime-conditioned backtest evaluation cannot be calculated."""


@dataclass(frozen=True, slots=True)
class RegimeBacktestEvaluation:
    """Container for regime-conditioned strategy evaluation."""

    regime_return_frame: pd.DataFrame
    metric_summary: pd.DataFrame
    metric_deltas: pd.DataFrame


def build_regime_strategy_return_frame(
    strategy_returns: Mapping[str, pd.Series],
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """Align strategy returns with regime labels."""
    return_frame = _build_strategy_return_frame(strategy_returns)
    labels = _validate_regime_labels(regime_labels)

    overlapping_dates = return_frame.index.intersection(labels.index)

    if overlapping_dates.empty:
        raise RegimeBacktestEvaluationError(
            "Strategy returns and regime labels have no overlapping dates"
        )

    aligned_returns = return_frame.loc[overlapping_dates].sort_index()
    aligned_labels = labels.loc[overlapping_dates].sort_index()

    regime_return_frame = aligned_returns.copy()
    regime_return_frame["regime"] = aligned_labels.astype(int)
    regime_return_frame.index.name = "date"

    return regime_return_frame


def calculate_regime_strategy_metric_summary(
    strategy_returns: Mapping[str, pd.Series],
    regime_labels: pd.Series,
    risk_free_rate: float = 0.0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> pd.DataFrame:
    """Calculate risk metrics by regime and strategy."""
    regime_return_frame = build_regime_strategy_return_frame(
        strategy_returns=strategy_returns,
        regime_labels=regime_labels,
    )

    strategy_columns = [
        str(column) for column in regime_return_frame.columns if str(column) != "regime"
    ]

    rows: list[dict[str, object]] = []

    for regime in sorted(regime_return_frame["regime"].unique()):
        regime_frame = regime_return_frame[regime_return_frame["regime"] == regime]

        for strategy_name in strategy_columns:
            strategy_series = regime_frame[strategy_name]

            summary = summarize_risk_metrics(
                returns=strategy_series,
                risk_free_rate=risk_free_rate,
                confidence_level=confidence_level,
                annualization_factor=annualization_factor,
            )

            rows.append(
                {
                    "regime": int(regime),
                    "strategy": strategy_name,
                    "observation_count": int(len(strategy_series)),
                    **summary,
                }
            )

    return pd.DataFrame(rows).sort_values(["regime", "strategy"]).reset_index(drop=True)


def calculate_regime_metric_deltas(
    metric_summary: pd.DataFrame,
    benchmark_strategy: str = "static",
    candidate_strategy: str = "dynamic",
) -> pd.DataFrame:
    """Calculate candidate-minus-benchmark metric deltas within each regime."""
    _validate_metric_summary(metric_summary)

    benchmark = str(benchmark_strategy).strip()
    candidate = str(candidate_strategy).strip()

    if not benchmark:
        raise RegimeBacktestEvaluationError("Benchmark strategy must be non-empty")

    if not candidate:
        raise RegimeBacktestEvaluationError("Candidate strategy must be non-empty")

    metric_columns = [
        str(column)
        for column in metric_summary.columns
        if str(column) not in {"regime", "strategy", "observation_count"}
    ]

    rows: list[dict[str, object]] = []

    for regime in sorted(metric_summary["regime"].unique()):
        regime_summary = metric_summary[metric_summary["regime"] == regime]

        benchmark_rows = regime_summary[regime_summary["strategy"] == benchmark]
        candidate_rows = regime_summary[regime_summary["strategy"] == candidate]

        if benchmark_rows.empty:
            raise RegimeBacktestEvaluationError(
                f"Benchmark strategy not found for regime {int(regime)}: {benchmark}"
            )

        if candidate_rows.empty:
            raise RegimeBacktestEvaluationError(
                f"Candidate strategy not found for regime {int(regime)}: {candidate}"
            )

        benchmark_row = benchmark_rows.iloc[0]
        candidate_row = candidate_rows.iloc[0]

        for metric in metric_columns:
            benchmark_value = float(benchmark_row[metric])
            candidate_value = float(candidate_row[metric])
            absolute_delta = candidate_value - benchmark_value

            if benchmark_value == 0:
                relative_delta = float("nan")
            else:
                relative_delta = absolute_delta / abs(benchmark_value)

            rows.append(
                {
                    "regime": int(regime),
                    "metric": metric,
                    "benchmark_strategy": benchmark,
                    "candidate_strategy": candidate,
                    "benchmark_value": benchmark_value,
                    "candidate_value": candidate_value,
                    "absolute_delta": absolute_delta,
                    "relative_delta": relative_delta,
                }
            )

    return pd.DataFrame(rows)


def evaluate_backtest_by_regime(
    strategy_returns: Mapping[str, pd.Series],
    regime_labels: pd.Series,
    benchmark_strategy: str = "static",
    candidate_strategy: str = "dynamic",
    risk_free_rate: float = 0.0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> RegimeBacktestEvaluation:
    """Build full regime-conditioned strategy evaluation."""
    regime_return_frame = build_regime_strategy_return_frame(
        strategy_returns=strategy_returns,
        regime_labels=regime_labels,
    )
    metric_summary = calculate_regime_strategy_metric_summary(
        strategy_returns=strategy_returns,
        regime_labels=regime_labels,
        risk_free_rate=risk_free_rate,
        confidence_level=confidence_level,
        annualization_factor=annualization_factor,
    )
    metric_deltas = calculate_regime_metric_deltas(
        metric_summary=metric_summary,
        benchmark_strategy=benchmark_strategy,
        candidate_strategy=candidate_strategy,
    )

    return RegimeBacktestEvaluation(
        regime_return_frame=regime_return_frame,
        metric_summary=metric_summary,
        metric_deltas=metric_deltas,
    )


def _build_strategy_return_frame(
    strategy_returns: Mapping[str, pd.Series],
) -> pd.DataFrame:
    if not strategy_returns:
        raise RegimeBacktestEvaluationError(
            "At least one strategy return series is required"
        )

    cleaned_returns: dict[str, pd.Series] = {}

    for strategy_name, returns in strategy_returns.items():
        clean_name = str(strategy_name).strip()

        if not clean_name:
            raise RegimeBacktestEvaluationError("Strategy names must be non-empty")

        if clean_name in cleaned_returns:
            raise RegimeBacktestEvaluationError(
                f"Duplicate strategy name: {clean_name}"
            )

        cleaned_returns[clean_name] = _validate_strategy_returns(
            strategy_name=clean_name,
            returns=returns,
        )

    return_frame = pd.concat(cleaned_returns.values(), axis=1, join="inner")

    if return_frame.empty:
        raise RegimeBacktestEvaluationError(
            "Strategy return series have no overlapping dates"
        )

    return_frame.index.name = "date"

    return return_frame.sort_index()


def _validate_strategy_returns(
    strategy_name: str,
    returns: pd.Series,
) -> pd.Series:
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise RegimeBacktestEvaluationError(
            f"Return series for {strategy_name} must use a DatetimeIndex"
        )

    if returns.empty:
        raise RegimeBacktestEvaluationError(
            f"Return series for {strategy_name} cannot be empty"
        )

    if returns.index.has_duplicates:
        raise RegimeBacktestEvaluationError(
            f"Return series for {strategy_name} contains duplicate dates"
        )

    clean_returns = pd.to_numeric(returns, errors="coerce")

    if clean_returns.isna().any():
        raise RegimeBacktestEvaluationError(
            f"Return series for {strategy_name} contains missing values"
        )

    return pd.Series(
        clean_returns,
        index=returns.index,
        name=strategy_name,
    ).sort_index()


def _validate_regime_labels(regime_labels: pd.Series) -> pd.Series:
    if not isinstance(regime_labels.index, pd.DatetimeIndex):
        raise RegimeBacktestEvaluationError(
            "Regime labels index must be a DatetimeIndex"
        )

    if regime_labels.empty:
        raise RegimeBacktestEvaluationError("Regime labels are empty")

    if regime_labels.index.has_duplicates:
        raise RegimeBacktestEvaluationError("Regime labels contain duplicate dates")

    numeric_labels = pd.to_numeric(regime_labels, errors="coerce")

    if numeric_labels.isna().any():
        raise RegimeBacktestEvaluationError(
            "Regime labels contain missing or non-numeric values"
        )

    labels = pd.Series(
        numeric_labels.astype(int),
        index=regime_labels.index,
        name="regime",
    )
    labels.index = pd.to_datetime(labels.index)

    return labels.sort_index()


def _validate_metric_summary(metric_summary: pd.DataFrame) -> None:
    required_columns = {"regime", "strategy", "observation_count"}
    missing_columns = required_columns.difference(metric_summary.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RegimeBacktestEvaluationError(
            f"Missing required metric summary column(s): {missing}"
        )

    if metric_summary.empty:
        raise RegimeBacktestEvaluationError("Metric summary cannot be empty")

    if metric_summary[["regime", "strategy"]].duplicated().any():
        raise RegimeBacktestEvaluationError(
            "Metric summary contains duplicate regime/strategy rows"
        )

    metric_columns = [
        str(column)
        for column in metric_summary.columns
        if str(column) not in {"regime", "strategy", "observation_count"}
    ]

    if not metric_columns:
        raise RegimeBacktestEvaluationError(
            "Metric summary must contain at least one metric column"
        )
