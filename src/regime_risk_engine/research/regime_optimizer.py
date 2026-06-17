from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd


class RegimePortfolioOptimizerError(ValueError):
    """Raised when regime portfolio optimization cannot be completed."""


@dataclass(frozen=True, slots=True)
class RegimePortfolioOptimizerConfig:
    """Configuration for regime-aware portfolio optimization."""

    min_weight: float = 0.0
    max_weight: float = 0.80
    risk_aversion: float = 3.0
    cvar_penalty: float = 2.0
    turnover_penalty: float = 0.25
    cvar_confidence_level: float = 0.95
    annualization_factor: int = 252
    n_candidate_portfolios: int = 2_000
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class RegimePortfolioOptimizationResult:
    """Optimized regime portfolio weights and diagnostics."""

    weight_table: pd.DataFrame
    diagnostics: pd.DataFrame


def optimize_regime_portfolios(
    asset_returns: pd.DataFrame,
    regime_labels: pd.Series,
    benchmark_weights: Mapping[str, float] | None = None,
    config: RegimePortfolioOptimizerConfig | None = None,
) -> RegimePortfolioOptimizationResult:
    """Optimize one constrained portfolio per detected market regime."""
    optimizer_config = config or RegimePortfolioOptimizerConfig()

    _validate_config(optimizer_config)
    _validate_inputs(asset_returns=asset_returns, regime_labels=regime_labels)

    aligned_returns, aligned_regimes = _align_returns_and_regimes(
        asset_returns=asset_returns,
        regime_labels=regime_labels,
    )

    tickers = [str(column) for column in aligned_returns.columns]
    clean_benchmark_weights = _resolve_benchmark_weights(
        benchmark_weights=benchmark_weights,
        tickers=tickers,
    )

    rng = np.random.default_rng(optimizer_config.random_state)

    weight_rows: list[dict[str, float | int]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []

    for regime, regime_returns in aligned_returns.groupby(aligned_regimes, sort=True):
        regime_id = int(regime)
        optimization = _optimize_single_regime(
            regime_returns=regime_returns,
            tickers=tickers,
            benchmark_weights=clean_benchmark_weights,
            config=optimizer_config,
            rng=rng,
        )

        weights = _extract_weight_array(optimization["weights"])

        weight_rows.append(
            {
                "regime": regime_id,
                **{
                    ticker: float(weight)
                    for ticker, weight in zip(tickers, weights, strict=True)
                },
            }
        )

        diagnostic_rows.append(
            {
                "regime": regime_id,
                "observation_count": int(len(regime_returns)),
                "expected_return": _extract_float_metric(
                    optimization,
                    "expected_return",
                ),
                "annualized_volatility": _extract_float_metric(
                    optimization,
                    "annualized_volatility",
                ),
                "sharpe_ratio": _extract_float_metric(
                    optimization,
                    "sharpe_ratio",
                ),
                "historical_cvar": _extract_float_metric(
                    optimization,
                    "historical_cvar",
                ),
                "turnover_from_benchmark": _extract_float_metric(
                    optimization,
                    "turnover_from_benchmark",
                ),
                "objective_score": _extract_float_metric(
                    optimization,
                    "objective_score",
                ),
                "largest_weight_asset": str(optimization["largest_weight_asset"]),
            }
        )

    weight_table = (
        pd.DataFrame(weight_rows).sort_values("regime").reset_index(drop=True)
    )
    diagnostics = (
        pd.DataFrame(diagnostic_rows).sort_values("regime").reset_index(drop=True)
    )

    return RegimePortfolioOptimizationResult(
        weight_table=weight_table,
        diagnostics=diagnostics,
    )


def _validate_config(config: RegimePortfolioOptimizerConfig) -> None:
    if config.min_weight < 0.0:
        raise RegimePortfolioOptimizerError("Minimum weight cannot be negative")

    if config.max_weight <= 0.0:
        raise RegimePortfolioOptimizerError("Maximum weight must be positive")

    if config.min_weight > config.max_weight:
        raise RegimePortfolioOptimizerError(
            "Minimum weight cannot exceed maximum weight"
        )

    if config.risk_aversion < 0.0:
        raise RegimePortfolioOptimizerError("Risk aversion cannot be negative")

    if config.cvar_penalty < 0.0:
        raise RegimePortfolioOptimizerError("CVaR penalty cannot be negative")

    if config.turnover_penalty < 0.0:
        raise RegimePortfolioOptimizerError("Turnover penalty cannot be negative")

    if not 0.0 < config.cvar_confidence_level < 1.0:
        raise RegimePortfolioOptimizerError(
            "CVaR confidence level must be between 0 and 1"
        )

    if config.annualization_factor <= 0:
        raise RegimePortfolioOptimizerError("Annualization factor must be positive")

    if config.n_candidate_portfolios <= 0:
        raise RegimePortfolioOptimizerError(
            "Number of candidate portfolios must be positive"
        )


def _validate_inputs(asset_returns: pd.DataFrame, regime_labels: pd.Series) -> None:
    if asset_returns.empty:
        raise RegimePortfolioOptimizerError("Asset returns cannot be empty")

    if regime_labels.empty:
        raise RegimePortfolioOptimizerError("Regime labels cannot be empty")

    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise RegimePortfolioOptimizerError(
            "Asset returns index must be a DatetimeIndex"
        )

    if not isinstance(regime_labels.index, pd.DatetimeIndex):
        raise RegimePortfolioOptimizerError(
            "Regime labels index must be a DatetimeIndex"
        )

    numeric_returns = asset_returns.apply(pd.to_numeric, errors="coerce")

    if numeric_returns.isna().any().any():
        raise RegimePortfolioOptimizerError(
            "Asset returns must be numeric and complete"
        )

    if regime_labels.isna().any():
        raise RegimePortfolioOptimizerError(
            "Regime labels cannot contain missing values"
        )

    if len(asset_returns.columns) < 2:
        raise RegimePortfolioOptimizerError("At least two assets are required")


def _align_returns_and_regimes(
    asset_returns: pd.DataFrame,
    regime_labels: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    common_index = asset_returns.index.intersection(regime_labels.index)

    if common_index.empty:
        raise RegimePortfolioOptimizerError(
            "Asset returns and regime labels have no overlapping dates"
        )

    aligned_returns = asset_returns.loc[common_index].astype(float).copy()
    aligned_regimes = regime_labels.loc[common_index].astype(int).copy()

    if aligned_regimes.nunique() < 2:
        raise RegimePortfolioOptimizerError("At least two regimes are required")

    return aligned_returns, aligned_regimes


def _resolve_benchmark_weights(
    benchmark_weights: Mapping[str, float] | None,
    tickers: list[str],
) -> np.ndarray:
    if benchmark_weights is None:
        return np.full(len(tickers), 1.0 / len(tickers))

    clean_weights = {
        str(ticker).strip().upper(): float(weight)
        for ticker, weight in benchmark_weights.items()
    }

    expected_tickers = {ticker.upper() for ticker in tickers}
    actual_tickers = set(clean_weights)

    if actual_tickers != expected_tickers:
        raise RegimePortfolioOptimizerError(
            f"Benchmark weights must contain exactly these tickers: "
            f"{sorted(expected_tickers)}"
        )

    weights = np.array(
        [clean_weights[ticker.upper()] for ticker in tickers],
        dtype=float,
    )

    if np.any(weights < 0.0):
        raise RegimePortfolioOptimizerError(
            "Benchmark weights cannot contain negative values"
        )

    if not np.isclose(float(weights.sum()), 1.0):
        raise RegimePortfolioOptimizerError("Benchmark weights must sum to 1.0")

    return weights


def _optimize_single_regime(
    regime_returns: pd.DataFrame,
    tickers: list[str],
    benchmark_weights: np.ndarray,
    config: RegimePortfolioOptimizerConfig,
    rng: np.random.Generator,
) -> dict[str, object]:
    candidate_weights = _generate_candidate_weights(
        regime_returns=regime_returns,
        config=config,
        rng=rng,
    )

    best_score = -np.inf
    best_weights: np.ndarray | None = None
    best_metrics: dict[str, float] | None = None

    for weights in candidate_weights:
        metrics = _calculate_portfolio_metrics(
            regime_returns=regime_returns,
            weights=weights,
            benchmark_weights=benchmark_weights,
            config=config,
        )
        score = _calculate_objective_score(
            metrics=metrics,
            config=config,
        )

        if score > best_score:
            best_score = score
            best_weights = weights
            best_metrics = metrics

    if best_weights is None or best_metrics is None:
        raise RegimePortfolioOptimizerError("No feasible portfolio was found")

    largest_weight_asset = tickers[int(np.argmax(best_weights))]

    return {
        "weights": best_weights,
        "largest_weight_asset": largest_weight_asset,
        "objective_score": best_score,
        **best_metrics,
    }


def _generate_candidate_weights(
    regime_returns: pd.DataFrame,
    config: RegimePortfolioOptimizerConfig,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    asset_count = len(regime_returns.columns)
    candidates: list[np.ndarray] = []

    candidates.append(
        _project_weights_to_constraints(
            np.full(asset_count, 1.0 / asset_count),
            min_weight=config.min_weight,
            max_weight=config.max_weight,
        )
    )

    candidates.append(
        _project_weights_to_constraints(
            _inverse_volatility_weights(regime_returns),
            min_weight=config.min_weight,
            max_weight=config.max_weight,
        )
    )

    candidates.append(
        _project_weights_to_constraints(
            _return_tilt_weights(regime_returns),
            min_weight=config.min_weight,
            max_weight=config.max_weight,
        )
    )

    random_candidates = rng.dirichlet(
        alpha=np.ones(asset_count),
        size=config.n_candidate_portfolios,
    )

    for candidate in random_candidates:
        projected_candidate = _project_weights_to_constraints(
            candidate,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
        )
        candidates.append(projected_candidate)

    return candidates


def _inverse_volatility_weights(regime_returns: pd.DataFrame) -> np.ndarray:
    volatility = regime_returns.std(ddof=1).replace(0.0, np.nan)
    inverse_volatility = 1.0 / volatility
    inverse_volatility = inverse_volatility.replace([np.inf, -np.inf], np.nan).fillna(
        0.0
    )

    inverse_volatility_sum = float(inverse_volatility.sum())

    if inverse_volatility_sum <= 0.0:
        return np.asarray(
            np.full(len(regime_returns.columns), 1.0 / len(regime_returns.columns)),
            dtype=float,
        )

    weights = np.asarray(inverse_volatility.to_numpy(dtype=float), dtype=float)

    return np.asarray(weights / inverse_volatility_sum, dtype=float)


def _return_tilt_weights(regime_returns: pd.DataFrame) -> np.ndarray:
    mean_returns = regime_returns.mean()
    mean_return_values = np.asarray(mean_returns.to_numpy(dtype=float), dtype=float)
    shifted_returns = mean_returns - float(mean_returns.min())
    shifted_return_sum = float(shifted_returns.sum())

    if shifted_return_sum <= 0.0:
        best_asset_position = int(np.argmax(mean_return_values))
        weights = np.zeros(len(mean_return_values), dtype=float)
        weights[best_asset_position] = 1.0

        return np.asarray(weights, dtype=float)

    shifted_values = np.asarray(shifted_returns.to_numpy(dtype=float), dtype=float)

    return np.asarray(shifted_values / shifted_return_sum, dtype=float)


def _project_weights_to_constraints(
    weights: np.ndarray,
    min_weight: float,
    max_weight: float,
) -> np.ndarray:
    asset_count = len(weights)

    if min_weight * asset_count > 1.0:
        raise RegimePortfolioOptimizerError(
            "Minimum weight is infeasible for number of assets"
        )

    if max_weight * asset_count < 1.0:
        raise RegimePortfolioOptimizerError(
            "Maximum weight is infeasible for number of assets"
        )

    projected = np.asarray(
        np.clip(weights.astype(float), min_weight, max_weight),
        dtype=float,
    )

    for _ in range(100):
        weight_sum = float(projected.sum())

        if np.isclose(weight_sum, 1.0):
            return np.asarray(projected / weight_sum, dtype=float)

        difference = 1.0 - weight_sum

        if difference > 0.0:
            available = projected < max_weight
            if not np.any(available):
                break
            projected[available] += difference / float(available.sum())
            projected = np.asarray(np.minimum(projected, max_weight), dtype=float)
        else:
            available = projected > min_weight
            if not np.any(available):
                break
            projected[available] += difference / float(available.sum())
            projected = np.asarray(np.maximum(projected, min_weight), dtype=float)

    final_sum = float(projected.sum())

    if not np.isclose(final_sum, 1.0):
        raise RegimePortfolioOptimizerError("Could not project weights to constraints")

    return np.asarray(projected / final_sum, dtype=float)


def _calculate_portfolio_metrics(
    regime_returns: pd.DataFrame,
    weights: np.ndarray,
    benchmark_weights: np.ndarray,
    config: RegimePortfolioOptimizerConfig,
) -> dict[str, float]:
    portfolio_returns = regime_returns.to_numpy(dtype=float) @ weights

    expected_return = float(np.mean(portfolio_returns) * config.annualization_factor)

    if len(portfolio_returns) < 2:
        annualized_volatility = 0.0
    else:
        annualized_volatility = float(
            np.std(portfolio_returns, ddof=1) * sqrt(config.annualization_factor)
        )

    if annualized_volatility == 0.0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float(expected_return / annualized_volatility)

    historical_cvar = _calculate_historical_cvar(
        portfolio_returns=portfolio_returns,
        confidence_level=config.cvar_confidence_level,
    )
    turnover_from_benchmark = float(np.abs(weights - benchmark_weights).sum())

    return {
        "expected_return": expected_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "historical_cvar": historical_cvar,
        "turnover_from_benchmark": turnover_from_benchmark,
    }


def _calculate_historical_cvar(
    portfolio_returns: np.ndarray,
    confidence_level: float,
) -> float:
    losses = -portfolio_returns
    threshold = float(np.quantile(losses, confidence_level))
    tail_losses = losses[losses >= threshold]

    if len(tail_losses) == 0:
        return 0.0

    return float(np.mean(tail_losses))


def _calculate_objective_score(
    metrics: dict[str, float],
    config: RegimePortfolioOptimizerConfig,
) -> float:
    expected_return = metrics["expected_return"]
    annualized_volatility = metrics["annualized_volatility"]
    historical_cvar = metrics["historical_cvar"]
    turnover_from_benchmark = metrics["turnover_from_benchmark"]

    return float(
        expected_return
        - config.risk_aversion * annualized_volatility**2
        - config.cvar_penalty * historical_cvar
        - config.turnover_penalty * turnover_from_benchmark
    )


def _extract_weight_array(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise RegimePortfolioOptimizerError("Expected optimized weights array")

    weights = np.asarray(value, dtype=float)

    if weights.ndim != 1:
        raise RegimePortfolioOptimizerError("Optimized weights must be one-dimensional")

    return weights


def _extract_float_metric(data: Mapping[str, object], key: str) -> float:
    numeric = pd.to_numeric(pd.Series([data[key]]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise RegimePortfolioOptimizerError(f"Expected numeric optimizer metric: {key}")

    return float(numeric)
