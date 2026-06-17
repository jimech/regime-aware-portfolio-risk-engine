from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from regime_risk_engine.research.regime_optimizer import (
    RegimePortfolioOptimizerConfig,
    optimize_regime_portfolios,
)


class WalkForwardRegimeOptimizerError(ValueError):
    """Raised when walk-forward regime optimization cannot be completed."""


@dataclass(frozen=True, slots=True)
class WalkForwardRegimeOptimizerConfig:
    """Configuration for walk-forward regime optimization."""

    train_window: int = 252
    test_window: int = 21
    min_training_observations: int = 30
    annualization_factor: int = 252


@dataclass(frozen=True, slots=True)
class WalkForwardRegimeOptimizationResult:
    """Walk-forward optimized allocation result."""

    return_frame: pd.DataFrame
    dynamic_weight_frame: pd.DataFrame
    optimization_diagnostics: pd.DataFrame
    metric_summary: pd.DataFrame
    metric_deltas: pd.DataFrame


def run_walk_forward_regime_optimization(
    asset_returns: pd.DataFrame,
    regime_labels: pd.Series,
    benchmark_weights: Mapping[str, float],
    config: WalkForwardRegimeOptimizerConfig | None = None,
    optimizer_config: RegimePortfolioOptimizerConfig | None = None,
) -> WalkForwardRegimeOptimizationResult:
    """Run walk-forward regime-aware portfolio optimization."""
    walk_config = config or WalkForwardRegimeOptimizerConfig()

    _validate_config(walk_config)
    _validate_inputs(
        asset_returns=asset_returns,
        regime_labels=regime_labels,
    )

    aligned_returns, aligned_regimes = _align_returns_and_regimes(
        asset_returns=asset_returns,
        regime_labels=regime_labels,
    )
    tickers = [str(column) for column in aligned_returns.columns]
    clean_benchmark_weights = _validate_benchmark_weights(
        benchmark_weights=benchmark_weights,
        tickers=tickers,
    )

    dynamic_weight_rows: list[pd.DataFrame] = []
    return_rows: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []

    window_number = 0

    for train_start, train_end, test_start, test_end in _iter_walk_forward_windows(
        observation_count=len(aligned_returns),
        train_window=walk_config.train_window,
        test_window=walk_config.test_window,
    ):
        train_returns = aligned_returns.iloc[train_start:train_end]
        train_regimes = aligned_regimes.iloc[train_start:train_end]
        test_returns = aligned_returns.iloc[test_start:test_end]
        test_regimes = aligned_regimes.iloc[test_start:test_end]

        if len(train_returns) < walk_config.min_training_observations:
            continue

        policy, used_optimizer = _fit_or_fallback_policy(
            train_returns=train_returns,
            train_regimes=train_regimes,
            benchmark_weights=clean_benchmark_weights,
            optimizer_config=optimizer_config,
        )

        test_weights = _build_test_weight_frame(
            test_regimes=test_regimes,
            policy=policy,
            benchmark_weights=clean_benchmark_weights,
            tickers=tickers,
        )

        static_returns = _calculate_weighted_returns(
            asset_returns=test_returns,
            weights=_build_constant_weight_frame(
                dates=test_returns.index,
                weights=clean_benchmark_weights,
                tickers=tickers,
            ),
        )
        dynamic_returns = _calculate_weighted_returns(
            asset_returns=test_returns,
            weights=test_weights,
        )

        return_frame = pd.DataFrame(
            {
                "static": static_returns,
                "dynamic": dynamic_returns,
                "regime": test_regimes.astype(int),
                "window": window_number,
            },
            index=test_returns.index,
        )

        dynamic_weight_rows.append(test_weights)
        return_rows.append(return_frame)
        diagnostic_rows.append(
            {
                "window": window_number,
                "train_start": train_returns.index.min(),
                "train_end": train_returns.index.max(),
                "test_start": test_returns.index.min(),
                "test_end": test_returns.index.max(),
                "training_observations": int(len(train_returns)),
                "test_observations": int(len(test_returns)),
                "training_regime_count": int(train_regimes.nunique()),
                "used_optimizer": used_optimizer,
            }
        )

        window_number += 1

    if not return_rows or not dynamic_weight_rows:
        raise WalkForwardRegimeOptimizerError(
            "No walk-forward windows were created. Use more data or smaller windows."
        )

    return_frame = pd.concat(return_rows, axis=0).sort_index()
    dynamic_weight_frame = pd.concat(dynamic_weight_rows, axis=0).sort_index()
    optimization_diagnostics = pd.DataFrame(diagnostic_rows)

    metric_summary = _build_metric_summary(
        return_frame=return_frame[["static", "dynamic"]],
        annualization_factor=walk_config.annualization_factor,
    )
    metric_deltas = _build_metric_deltas(
        metric_summary=metric_summary,
        benchmark_strategy="static",
        candidate_strategy="dynamic",
    )

    return WalkForwardRegimeOptimizationResult(
        return_frame=return_frame,
        dynamic_weight_frame=dynamic_weight_frame,
        optimization_diagnostics=optimization_diagnostics,
        metric_summary=metric_summary,
        metric_deltas=metric_deltas,
    )


def _validate_config(config: WalkForwardRegimeOptimizerConfig) -> None:
    if config.train_window <= 0:
        raise WalkForwardRegimeOptimizerError("Train window must be positive")

    if config.test_window <= 0:
        raise WalkForwardRegimeOptimizerError("Test window must be positive")

    if config.min_training_observations <= 0:
        raise WalkForwardRegimeOptimizerError(
            "Minimum training observations must be positive"
        )

    if config.min_training_observations > config.train_window:
        raise WalkForwardRegimeOptimizerError(
            "Minimum training observations cannot exceed train window"
        )

    if config.annualization_factor <= 0:
        raise WalkForwardRegimeOptimizerError("Annualization factor must be positive")


def _validate_inputs(asset_returns: pd.DataFrame, regime_labels: pd.Series) -> None:
    if asset_returns.empty:
        raise WalkForwardRegimeOptimizerError("Asset returns cannot be empty")

    if regime_labels.empty:
        raise WalkForwardRegimeOptimizerError("Regime labels cannot be empty")

    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise WalkForwardRegimeOptimizerError(
            "Asset returns index must be a DatetimeIndex"
        )

    if not isinstance(regime_labels.index, pd.DatetimeIndex):
        raise WalkForwardRegimeOptimizerError(
            "Regime labels index must be a DatetimeIndex"
        )

    numeric_returns = asset_returns.apply(pd.to_numeric, errors="coerce")

    if numeric_returns.isna().any().any():
        raise WalkForwardRegimeOptimizerError(
            "Asset returns must be numeric and complete"
        )

    if regime_labels.isna().any():
        raise WalkForwardRegimeOptimizerError(
            "Regime labels cannot contain missing values"
        )

    if len(asset_returns.columns) < 2:
        raise WalkForwardRegimeOptimizerError("At least two assets are required")


def _align_returns_and_regimes(
    asset_returns: pd.DataFrame,
    regime_labels: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    common_index = asset_returns.index.intersection(regime_labels.index)

    if common_index.empty:
        raise WalkForwardRegimeOptimizerError(
            "Asset returns and regime labels have no overlapping dates"
        )

    aligned_returns = asset_returns.loc[common_index].astype(float).copy()
    aligned_regimes = regime_labels.loc[common_index].astype(int).copy()

    if aligned_regimes.nunique() < 2:
        raise WalkForwardRegimeOptimizerError("At least two regimes are required")

    return aligned_returns, aligned_regimes


def _validate_benchmark_weights(
    benchmark_weights: Mapping[str, float],
    tickers: list[str],
) -> dict[str, float]:
    if not benchmark_weights:
        raise WalkForwardRegimeOptimizerError("Benchmark weights cannot be empty")

    clean_weights = {
        str(ticker).strip().upper(): float(weight)
        for ticker, weight in benchmark_weights.items()
    }

    expected_tickers = {ticker.upper() for ticker in tickers}
    actual_tickers = set(clean_weights)

    if actual_tickers != expected_tickers:
        raise WalkForwardRegimeOptimizerError(
            f"Benchmark weights must contain exactly these tickers: "
            f"{sorted(expected_tickers)}"
        )

    if any(weight < 0.0 for weight in clean_weights.values()):
        raise WalkForwardRegimeOptimizerError(
            "Benchmark weights cannot contain negative values"
        )

    if not np.isclose(sum(clean_weights.values()), 1.0):
        raise WalkForwardRegimeOptimizerError("Benchmark weights must sum to 1.0")

    return clean_weights


def _iter_walk_forward_windows(
    observation_count: int,
    train_window: int,
    test_window: int,
) -> list[tuple[int, int, int, int]]:
    windows: list[tuple[int, int, int, int]] = []

    train_end = train_window

    while train_end < observation_count:
        train_start = train_end - train_window
        test_start = train_end
        test_end = min(test_start + test_window, observation_count)

        if test_start >= test_end:
            break

        windows.append((train_start, train_end, test_start, test_end))
        train_end += test_window

    return windows


def _fit_or_fallback_policy(
    train_returns: pd.DataFrame,
    train_regimes: pd.Series,
    benchmark_weights: Mapping[str, float],
    optimizer_config: RegimePortfolioOptimizerConfig | None,
) -> tuple[dict[int, dict[str, float]], bool]:
    if train_regimes.nunique() < 2:
        return _build_benchmark_policy(
            regimes=train_regimes.unique(),
            benchmark_weights=benchmark_weights,
        ), False

    optimization_result = optimize_regime_portfolios(
        asset_returns=train_returns,
        regime_labels=train_regimes,
        benchmark_weights=benchmark_weights,
        config=optimizer_config,
    )

    return _convert_weight_table_to_policy(optimization_result.weight_table), True


def _build_benchmark_policy(
    regimes: np.ndarray,
    benchmark_weights: Mapping[str, float],
) -> dict[int, dict[str, float]]:
    return {
        int(regime): {
            ticker: float(weight) for ticker, weight in benchmark_weights.items()
        }
        for regime in regimes
    }


def _convert_weight_table_to_policy(
    weight_table: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    if weight_table.empty:
        raise WalkForwardRegimeOptimizerError("Weight table cannot be empty")

    tickers = [str(column) for column in weight_table.columns if column != "regime"]
    policy: dict[int, dict[str, float]] = {}

    for _, row in weight_table.iterrows():
        regime = _to_int(row["regime"])
        weights = {ticker: _to_float(row[ticker]) for ticker in tickers}
        policy[regime] = weights

    return policy


def _build_test_weight_frame(
    test_regimes: pd.Series,
    policy: Mapping[int, Mapping[str, float]],
    benchmark_weights: Mapping[str, float],
    tickers: list[str],
) -> pd.DataFrame:
    rows = []

    for regime in test_regimes.astype(int):
        regime_policy = policy.get(int(regime), benchmark_weights)
        rows.append([float(regime_policy[ticker]) for ticker in tickers])

    return pd.DataFrame(
        rows,
        index=test_regimes.index,
        columns=tickers,
    )


def _build_constant_weight_frame(
    dates: pd.DatetimeIndex,
    weights: Mapping[str, float],
    tickers: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [[float(weights[ticker]) for ticker in tickers] for _ in range(len(dates))],
        index=dates,
        columns=tickers,
    )


def _calculate_weighted_returns(
    asset_returns: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.Series:
    weighted_returns = (asset_returns * weights).sum(axis=1)

    return pd.Series(weighted_returns, index=asset_returns.index)


def _build_metric_summary(
    return_frame: pd.DataFrame,
    annualization_factor: int,
) -> pd.DataFrame:
    rows = {
        strategy: _calculate_return_metrics(
            returns=return_frame[strategy],
            annualization_factor=annualization_factor,
        )
        for strategy in return_frame.columns
    }

    return pd.DataFrame.from_dict(rows, orient="index")


def _calculate_return_metrics(
    returns: pd.Series,
    annualization_factor: int,
) -> dict[str, float]:
    clean_returns = returns.dropna().astype(float)

    if clean_returns.empty:
        raise WalkForwardRegimeOptimizerError(
            "Cannot calculate metrics on empty returns"
        )

    cumulative_return = float((1.0 + clean_returns).prod() - 1.0)
    annualized_return = float(
        (1.0 + cumulative_return) ** (annualization_factor / len(clean_returns)) - 1.0
    )

    if len(clean_returns) < 2:
        annualized_volatility = 0.0
    else:
        annualized_volatility = float(
            clean_returns.std(ddof=1) * sqrt(annualization_factor)
        )

    if annualized_volatility == 0.0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float(annualized_return / annualized_volatility)

    wealth = (1.0 + clean_returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def _build_metric_deltas(
    metric_summary: pd.DataFrame,
    benchmark_strategy: str,
    candidate_strategy: str,
) -> pd.DataFrame:
    rows = []

    for metric in metric_summary.columns:
        benchmark_value = float(metric_summary.loc[benchmark_strategy, metric])
        candidate_value = float(metric_summary.loc[candidate_strategy, metric])
        absolute_delta = candidate_value - benchmark_value

        if benchmark_value == 0.0:
            relative_delta = 0.0
        else:
            relative_delta = absolute_delta / abs(benchmark_value)

        rows.append(
            {
                "metric": metric,
                "benchmark_strategy": benchmark_strategy,
                "candidate_strategy": candidate_strategy,
                "benchmark_value": benchmark_value,
                "candidate_value": candidate_value,
                "absolute_delta": absolute_delta,
                "relative_delta": relative_delta,
            }
        )

    return pd.DataFrame(rows)


def _to_int(value: object) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise WalkForwardRegimeOptimizerError("Expected integer regime value")

    return int(numeric)


def _to_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise WalkForwardRegimeOptimizerError("Expected numeric weight value")

    return float(numeric)
