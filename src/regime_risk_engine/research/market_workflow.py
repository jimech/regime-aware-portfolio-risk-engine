from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from regime_risk_engine.backtesting.comparison import StrategyComparisonResult
from regime_risk_engine.backtesting.engine import BacktestResult
from regime_risk_engine.backtesting.regime_evaluation import RegimeBacktestEvaluation
from regime_risk_engine.research.pipeline import (
    InvestmentResearchPipelineResult,
    run_investment_research_pipeline,
)


class MarketResearchWorkflowError(ValueError):
    """Raised when the market research workflow cannot be completed."""


@dataclass(frozen=True, slots=True)
class MarketResearchWorkflowConfig:
    """Configuration for the market research workflow."""

    n_regimes: int = 3
    feature_window: int = 5
    annualization_factor: int = 252
    transaction_cost_bps: float = 0.0
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class MarketResearchWorkflowResult:
    """Container for real market research workflow outputs."""

    asset_returns: pd.DataFrame
    feature_matrix: pd.DataFrame
    regime_labels: pd.Series
    static_weight_frame: pd.DataFrame
    dynamic_target_weight_frame: pd.DataFrame
    dynamic_applied_weight_frame: pd.DataFrame
    strategy_comparison: StrategyComparisonResult
    regime_evaluation: RegimeBacktestEvaluation
    research_result: InvestmentResearchPipelineResult


PRICE_COLUMNS = {"date", "ticker"}

PRICE_VALUE_CANDIDATES = [
    "adjusted_close",
    "adj_close",
    "close",
    "price",
]


def run_market_research_workflow(
    price_data: pd.DataFrame,
    static_weights: Mapping[str, float],
    regime_weight_policy: Mapping[int, Mapping[str, float]],
    config: MarketResearchWorkflowConfig | None = None,
) -> MarketResearchWorkflowResult:
    """Run the end-to-end market research workflow from price data."""
    workflow_config = config or MarketResearchWorkflowConfig()

    _validate_config(workflow_config)

    price_frame = _prepare_price_frame(price_data)
    tickers = list(price_frame.columns)

    clean_static_weights = _validate_weight_mapping(
        weights=static_weights,
        tickers=tickers,
        label="static weights",
    )
    clean_regime_policy = _validate_regime_policy(
        regime_weight_policy=regime_weight_policy,
        tickers=tickers,
        n_regimes=workflow_config.n_regimes,
    )

    asset_returns = price_frame.pct_change().dropna(how="any")
    _validate_return_frame(asset_returns)

    feature_matrix = _build_regime_feature_matrix(
        asset_returns=asset_returns,
        window=workflow_config.feature_window,
    )

    regime_labels = _fit_kmeans_regimes(
        feature_matrix=feature_matrix,
        n_regimes=workflow_config.n_regimes,
        random_state=workflow_config.random_state,
    )

    aligned_returns = asset_returns.loc[regime_labels.index].copy()

    static_weight_frame = _build_constant_weight_frame(
        dates=aligned_returns.index,
        weights=clean_static_weights,
        tickers=tickers,
    )
    dynamic_target_weight_frame = _build_dynamic_target_weight_frame(
        regime_labels=regime_labels,
        regime_policy=clean_regime_policy,
        tickers=tickers,
    )
    dynamic_applied_weight_frame = dynamic_target_weight_frame.shift(1).dropna(
        how="any"
    )

    common_dates = aligned_returns.index.intersection(
        dynamic_applied_weight_frame.index
    )

    if common_dates.empty:
        raise MarketResearchWorkflowError(
            "No common dates remain after applying one-period weight lag"
        )

    aligned_returns = aligned_returns.loc[common_dates]
    static_weight_frame = static_weight_frame.loc[common_dates]
    dynamic_applied_weight_frame = dynamic_applied_weight_frame.loc[common_dates]
    regime_labels = regime_labels.loc[common_dates]

    static_backtest = _run_weighted_backtest(
        asset_returns=aligned_returns,
        applied_weights=static_weight_frame,
        transaction_cost_bps=workflow_config.transaction_cost_bps,
        strategy_name="static",
    )
    dynamic_backtest = _run_weighted_backtest(
        asset_returns=aligned_returns,
        applied_weights=dynamic_applied_weight_frame,
        transaction_cost_bps=workflow_config.transaction_cost_bps,
        strategy_name="dynamic",
    )

    strategy_comparison = _build_strategy_comparison(
        static_backtest=static_backtest,
        dynamic_backtest=dynamic_backtest,
        annualization_factor=workflow_config.annualization_factor,
    )
    regime_evaluation = _build_regime_evaluation(
        return_comparison=strategy_comparison.return_comparison,
        regime_labels=regime_labels,
        annualization_factor=workflow_config.annualization_factor,
    )
    research_result = run_investment_research_pipeline(
        strategy_comparison=strategy_comparison,
        regime_evaluation=regime_evaluation,
        benchmark_strategy="static",
        candidate_strategy="dynamic",
        primary_regime_metric="sharpe_ratio",
    )

    return MarketResearchWorkflowResult(
        asset_returns=aligned_returns,
        feature_matrix=feature_matrix,
        regime_labels=regime_labels,
        static_weight_frame=static_weight_frame,
        dynamic_target_weight_frame=dynamic_target_weight_frame,
        dynamic_applied_weight_frame=dynamic_applied_weight_frame,
        strategy_comparison=strategy_comparison,
        regime_evaluation=regime_evaluation,
        research_result=research_result,
    )


def _validate_config(config: MarketResearchWorkflowConfig) -> None:
    if config.n_regimes < 2:
        raise MarketResearchWorkflowError("At least two regimes are required")

    if config.feature_window < 2:
        raise MarketResearchWorkflowError("Feature window must be at least two")

    if config.annualization_factor <= 0:
        raise MarketResearchWorkflowError("Annualization factor must be positive")

    if config.transaction_cost_bps < 0:
        raise MarketResearchWorkflowError("Transaction costs cannot be negative")


def _prepare_price_frame(price_data: pd.DataFrame) -> pd.DataFrame:
    if price_data.empty:
        raise MarketResearchWorkflowError("Price data cannot be empty")

    missing_columns = PRICE_COLUMNS.difference(price_data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise MarketResearchWorkflowError(f"Missing price column(s): {missing}")

    price_column = _resolve_price_column(price_data)

    prices = price_data.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.strip().str.upper()
    prices[price_column] = pd.to_numeric(prices[price_column], errors="coerce")

    if prices["date"].isna().any():
        raise MarketResearchWorkflowError("Price data contains invalid dates")

    if prices["ticker"].eq("").any():
        raise MarketResearchWorkflowError("Price data contains empty tickers")

    if prices[price_column].isna().any():
        raise MarketResearchWorkflowError("Price data contains missing prices")

    if (prices[price_column] <= 0).any():
        raise MarketResearchWorkflowError("Price data contains non-positive prices")

    if prices[["date", "ticker"]].duplicated().any():
        raise MarketResearchWorkflowError(
            "Price data contains duplicate date/ticker rows"
        )

    price_frame = prices.pivot(
        index="date",
        columns="ticker",
        values=price_column,
    ).sort_index()

    if price_frame.isna().any().any():
        raise MarketResearchWorkflowError(
            "Price data must contain a complete date/ticker panel"
        )

    if len(price_frame.columns) < 2:
        raise MarketResearchWorkflowError("At least two tickers are required")

    return price_frame


def _resolve_price_column(price_data: pd.DataFrame) -> str:
    for column in PRICE_VALUE_CANDIDATES:
        if column in price_data.columns:
            return column

    candidates = ", ".join(PRICE_VALUE_CANDIDATES)
    raise MarketResearchWorkflowError(
        f"Price data must contain one price column from: {candidates}"
    )


def _validate_return_frame(asset_returns: pd.DataFrame) -> None:
    if asset_returns.empty:
        raise MarketResearchWorkflowError("Asset returns cannot be empty")

    if asset_returns.isna().any().any():
        raise MarketResearchWorkflowError("Asset returns contain missing values")

    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise MarketResearchWorkflowError("Asset returns index must be a DatetimeIndex")


def _build_regime_feature_matrix(
    asset_returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    rolling_mean = asset_returns.rolling(window).mean()
    rolling_volatility = asset_returns.rolling(window).std(ddof=1)
    rolling_correlation = _calculate_rolling_average_correlation(
        asset_returns=asset_returns,
        window=window,
    )

    feature_matrix = pd.concat(
        [
            rolling_mean.add_suffix("_rolling_mean"),
            rolling_volatility.add_suffix("_rolling_volatility"),
            rolling_correlation.rename("average_correlation"),
        ],
        axis=1,
    )

    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
    feature_matrix = feature_matrix.fillna(0.0)

    feature_matrix = feature_matrix.loc[asset_returns.index[window - 1 :]]

    if feature_matrix.empty:
        raise MarketResearchWorkflowError(
            "Feature matrix is empty. Use more price history or a smaller window."
        )

    return feature_matrix


def _calculate_rolling_average_correlation(
    asset_returns: pd.DataFrame,
    window: int,
) -> pd.Series:
    values: list[float] = []
    dates: list[pd.Timestamp] = []

    for end_position in range(window, len(asset_returns) + 1):
        window_frame = asset_returns.iloc[end_position - window : end_position]
        correlation_matrix = window_frame.corr().to_numpy(dtype=float)
        mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
        off_diagonal_values = correlation_matrix[mask]
        valid_correlations = off_diagonal_values[~np.isnan(off_diagonal_values)]

        if valid_correlations.size == 0:
            average_correlation = 0.0
        else:
            average_correlation = float(np.mean(valid_correlations))

        values.append(average_correlation)
        dates.append(pd.Timestamp(asset_returns.index[end_position - 1]))

    return pd.Series(values, index=pd.DatetimeIndex(dates))


def _fit_kmeans_regimes(
    feature_matrix: pd.DataFrame,
    n_regimes: int,
    random_state: int,
) -> pd.Series:
    if len(feature_matrix) < n_regimes:
        raise MarketResearchWorkflowError(
            "Feature matrix must have at least as many rows as regimes"
        )

    model = KMeans(
        n_clusters=n_regimes,
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(feature_matrix)

    return pd.Series(
        labels.astype(int),
        index=feature_matrix.index,
        name="regime",
    )


def _validate_weight_mapping(
    weights: Mapping[str, float],
    tickers: list[str],
    label: str,
) -> dict[str, float]:
    if not weights:
        raise MarketResearchWorkflowError(f"{label} cannot be empty")

    clean_weights = {
        str(ticker).strip().upper(): float(weight) for ticker, weight in weights.items()
    }

    expected_tickers = set(tickers)
    actual_tickers = set(clean_weights)

    if actual_tickers != expected_tickers:
        raise MarketResearchWorkflowError(
            f"{label} must contain exactly these tickers: {sorted(expected_tickers)}"
        )

    if any(weight < 0 for weight in clean_weights.values()):
        raise MarketResearchWorkflowError(f"{label} cannot contain negative weights")

    total_weight = sum(clean_weights.values())

    if not np.isclose(total_weight, 1.0):
        raise MarketResearchWorkflowError(f"{label} must sum to 1.0")

    return clean_weights


def _validate_regime_policy(
    regime_weight_policy: Mapping[int, Mapping[str, float]],
    tickers: list[str],
    n_regimes: int,
) -> dict[int, dict[str, float]]:
    expected_regimes = set(range(n_regimes))
    actual_regimes = {int(regime) for regime in regime_weight_policy}

    if actual_regimes != expected_regimes:
        raise MarketResearchWorkflowError(
            f"Regime policy must contain regimes: {sorted(expected_regimes)}"
        )

    return {
        int(regime): _validate_weight_mapping(
            weights=weights,
            tickers=tickers,
            label=f"regime {regime} weights",
        )
        for regime, weights in regime_weight_policy.items()
    }


def _build_constant_weight_frame(
    dates: pd.DatetimeIndex,
    weights: Mapping[str, float],
    tickers: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        [[weights[ticker] for ticker in tickers] for _ in range(len(dates))],
        index=dates,
        columns=tickers,
    )


def _build_dynamic_target_weight_frame(
    regime_labels: pd.Series,
    regime_policy: Mapping[int, Mapping[str, float]],
    tickers: list[str],
) -> pd.DataFrame:
    rows = []

    for regime in regime_labels.astype(int):
        rows.append([regime_policy[int(regime)][ticker] for ticker in tickers])

    return pd.DataFrame(
        rows,
        index=regime_labels.index,
        columns=tickers,
    )


def _run_weighted_backtest(
    asset_returns: pd.DataFrame,
    applied_weights: pd.DataFrame,
    transaction_cost_bps: float,
    strategy_name: str,
) -> BacktestResult:
    gross_returns = (asset_returns * applied_weights).sum(axis=1)
    gross_returns.name = strategy_name

    turnover = applied_weights.diff().abs().sum(axis=1).fillna(0.0)
    turnover.name = "turnover"

    transaction_costs = turnover * (transaction_cost_bps / 10_000.0)
    transaction_costs.name = "transaction_cost"

    net_returns = gross_returns - transaction_costs
    net_returns.name = strategy_name

    return BacktestResult(
        gross_returns=gross_returns,
        net_returns=net_returns,
        turnover=turnover,
        transaction_costs=transaction_costs,
        applied_weights=applied_weights,
    )


def _build_strategy_comparison(
    static_backtest: BacktestResult,
    dynamic_backtest: BacktestResult,
    annualization_factor: int,
) -> StrategyComparisonResult:
    return_comparison = pd.concat(
        [
            static_backtest.net_returns.rename("static"),
            dynamic_backtest.net_returns.rename("dynamic"),
        ],
        axis=1,
    )

    metric_summary = _build_strategy_metric_summary(
        return_comparison=return_comparison,
        annualization_factor=annualization_factor,
    )
    metric_deltas = _build_metric_deltas(
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


def _build_strategy_metric_summary(
    return_comparison: pd.DataFrame,
    annualization_factor: int,
) -> pd.DataFrame:
    rows = {
        strategy: _calculate_return_metrics(
            returns=return_comparison[strategy],
            annualization_factor=annualization_factor,
        )
        for strategy in return_comparison.columns
    }

    return pd.DataFrame.from_dict(rows, orient="index")


def _calculate_return_metrics(
    returns: pd.Series,
    annualization_factor: int,
) -> dict[str, float]:
    clean_returns = returns.dropna().astype(float)

    if clean_returns.empty:
        raise MarketResearchWorkflowError("Cannot calculate metrics on empty returns")

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


def _build_regime_evaluation(
    return_comparison: pd.DataFrame,
    regime_labels: pd.Series,
    annualization_factor: int,
) -> RegimeBacktestEvaluation:
    regime_return_frame = return_comparison.copy()
    regime_return_frame["regime"] = regime_labels.astype(int)

    metric_summary = _build_regime_metric_summary(
        regime_return_frame=regime_return_frame,
        annualization_factor=annualization_factor,
    )
    metric_deltas = _build_regime_metric_deltas(
        metric_summary=metric_summary,
        benchmark_strategy="static",
        candidate_strategy="dynamic",
    )

    return RegimeBacktestEvaluation(
        regime_return_frame=regime_return_frame,
        metric_summary=metric_summary,
        metric_deltas=metric_deltas,
    )


def _build_regime_metric_summary(
    regime_return_frame: pd.DataFrame,
    annualization_factor: int,
) -> pd.DataFrame:
    rows = []

    for regime, regime_frame in regime_return_frame.groupby("regime"):
        for strategy in ["static", "dynamic"]:
            metrics = _calculate_return_metrics(
                returns=regime_frame[strategy],
                annualization_factor=annualization_factor,
            )
            rows.append(
                {
                    "regime": int(regime),
                    "strategy": strategy,
                    "observation_count": int(len(regime_frame)),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def _build_regime_metric_deltas(
    metric_summary: pd.DataFrame,
    benchmark_strategy: str,
    candidate_strategy: str,
) -> pd.DataFrame:
    rows = []
    metrics = [
        column
        for column in metric_summary.columns
        if column not in {"regime", "strategy", "observation_count"}
    ]

    for regime, regime_frame in metric_summary.groupby("regime"):
        benchmark_row = regime_frame[
            regime_frame["strategy"] == benchmark_strategy
        ].iloc[0]
        candidate_row = regime_frame[
            regime_frame["strategy"] == candidate_strategy
        ].iloc[0]

        for metric in metrics:
            benchmark_value = float(benchmark_row[metric])
            candidate_value = float(candidate_row[metric])
            absolute_delta = candidate_value - benchmark_value

            if benchmark_value == 0.0:
                relative_delta = 0.0
            else:
                relative_delta = absolute_delta / abs(benchmark_value)

            rows.append(
                {
                    "regime": int(regime),
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
