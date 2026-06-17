from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.research.market_workflow import (
    MarketResearchWorkflowConfig,
    MarketResearchWorkflowError,
    MarketResearchWorkflowResult,
    run_market_research_workflow,
)
from regime_risk_engine.research.regime_optimizer import (
    RegimePortfolioOptimizationResult,
    RegimePortfolioOptimizerConfig,
    optimize_regime_portfolios,
)


class OptimizedMarketResearchWorkflowError(ValueError):
    """Raised when the optimized market research workflow cannot be completed."""


@dataclass(frozen=True, slots=True)
class OptimizedMarketResearchWorkflowResult:
    """Container for optimized market research workflow outputs."""

    initial_workflow_result: MarketResearchWorkflowResult
    optimization_result: RegimePortfolioOptimizationResult
    optimized_regime_policy: dict[int, dict[str, float]]
    optimized_workflow_result: MarketResearchWorkflowResult


def run_optimized_market_research_workflow(
    price_data: pd.DataFrame,
    static_weights: dict[str, float],
    market_config: MarketResearchWorkflowConfig | None = None,
    optimizer_config: RegimePortfolioOptimizerConfig | None = None,
) -> OptimizedMarketResearchWorkflowResult:
    """Run market research workflow with optimizer-learned regime weights."""
    clean_market_config = market_config or MarketResearchWorkflowConfig()
    tickers = _extract_tickers(price_data)

    seed_policy = _build_equal_weight_regime_policy(
        tickers=tickers,
        n_regimes=clean_market_config.n_regimes,
    )

    initial_result = run_market_research_workflow(
        price_data=price_data,
        static_weights=static_weights,
        regime_weight_policy=seed_policy,
        config=clean_market_config,
    )

    optimization_result = optimize_regime_portfolios(
        asset_returns=initial_result.asset_returns,
        regime_labels=initial_result.regime_labels,
        benchmark_weights=static_weights,
        config=optimizer_config,
    )

    optimized_policy = _convert_weight_table_to_policy(optimization_result.weight_table)

    optimized_result = run_market_research_workflow(
        price_data=price_data,
        static_weights=static_weights,
        regime_weight_policy=optimized_policy,
        config=clean_market_config,
    )

    return OptimizedMarketResearchWorkflowResult(
        initial_workflow_result=initial_result,
        optimization_result=optimization_result,
        optimized_regime_policy=optimized_policy,
        optimized_workflow_result=optimized_result,
    )


def _extract_tickers(price_data: pd.DataFrame) -> list[str]:
    if price_data.empty:
        raise OptimizedMarketResearchWorkflowError("Price data cannot be empty")

    if "ticker" not in price_data.columns:
        raise OptimizedMarketResearchWorkflowError(
            "Price data must contain a ticker column"
        )

    tickers = sorted(
        {
            str(ticker).strip().upper()
            for ticker in price_data["ticker"].dropna().unique()
        }
    )

    if len(tickers) < 2:
        raise OptimizedMarketResearchWorkflowError("At least two tickers are required")

    if any(not ticker for ticker in tickers):
        raise OptimizedMarketResearchWorkflowError("Price data contains empty tickers")

    return tickers


def _build_equal_weight_regime_policy(
    tickers: list[str],
    n_regimes: int,
) -> dict[int, dict[str, float]]:
    if n_regimes < 2:
        raise MarketResearchWorkflowError("At least two regimes are required")

    equal_weight = 1.0 / len(tickers)

    return {
        regime: {ticker: equal_weight for ticker in tickers}
        for regime in range(n_regimes)
    }


def _convert_weight_table_to_policy(
    weight_table: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    if weight_table.empty:
        raise OptimizedMarketResearchWorkflowError("Weight table cannot be empty")

    if "regime" not in weight_table.columns:
        raise OptimizedMarketResearchWorkflowError(
            "Weight table must contain a regime column"
        )

    tickers = [str(column) for column in weight_table.columns if column != "regime"]

    if not tickers:
        raise OptimizedMarketResearchWorkflowError(
            "Weight table must contain asset weight columns"
        )

    policy: dict[int, dict[str, float]] = {}

    for _, row in weight_table.iterrows():
        regime = _to_int(row["regime"])
        weights = {ticker: _to_float(row[ticker]) for ticker in tickers}
        total_weight = sum(weights.values())

        if not pd.Series([total_weight]).sub(1.0).abs().lt(1e-8).iloc[0]:
            raise OptimizedMarketResearchWorkflowError(
                f"Optimized weights for regime {regime} do not sum to 1.0"
            )

        policy[regime] = weights

    return policy


def _to_int(value: object) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise OptimizedMarketResearchWorkflowError("Expected integer regime value")

    return int(numeric)


def _to_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise OptimizedMarketResearchWorkflowError("Expected numeric weight value")

    return float(numeric)
