import pandas as pd
import pytest

from regime_risk_engine.research.regime_optimizer import (
    RegimePortfolioOptimizationResult,
    RegimePortfolioOptimizerConfig,
    RegimePortfolioOptimizerError,
    optimize_regime_portfolios,
)


def make_asset_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    rows = []

    for index, _date in enumerate(dates):
        if index < 30:
            rows.append(
                {
                    "SPY": 0.004,
                    "TLT": 0.0005,
                    "GLD": 0.0002,
                }
            )
        elif index < 60:
            rows.append(
                {
                    "SPY": -0.006,
                    "TLT": 0.003,
                    "GLD": 0.001,
                }
            )
        else:
            rows.append(
                {
                    "SPY": 0.0005,
                    "TLT": -0.001,
                    "GLD": 0.004,
                }
            )

    return pd.DataFrame(rows, index=dates)


def make_regime_labels() -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=90, freq="D")

    return pd.Series(
        [0] * 30 + [1] * 30 + [2] * 30,
        index=dates,
        name="regime",
    )


def make_benchmark_weights() -> dict[str, float]:
    return {
        "SPY": 0.60,
        "TLT": 0.30,
        "GLD": 0.10,
    }


def test_optimize_regime_portfolios() -> None:
    result = optimize_regime_portfolios(
        asset_returns=make_asset_returns(),
        regime_labels=make_regime_labels(),
        benchmark_weights=make_benchmark_weights(),
        config=RegimePortfolioOptimizerConfig(
            max_weight=0.80,
            risk_aversion=0.25,
            cvar_penalty=0.25,
            turnover_penalty=0.0,
            n_candidate_portfolios=500,
            random_state=7,
        ),
    )

    assert isinstance(result, RegimePortfolioOptimizationResult)
    assert len(result.weight_table) == 3
    assert len(result.diagnostics) == 3

    weight_columns = ["SPY", "TLT", "GLD"]
    weight_sums = result.weight_table[weight_columns].sum(axis=1)

    assert all(abs(weight_sum - 1.0) < 1e-10 for weight_sum in weight_sums)
    assert result.weight_table[weight_columns].max().max() <= 0.80 + 1e-10
    assert set(result.diagnostics["largest_weight_asset"]).issubset(
        {"SPY", "TLT", "GLD"}
    )


def test_optimizer_tilts_toward_regime_leaders() -> None:
    result = optimize_regime_portfolios(
        asset_returns=make_asset_returns(),
        regime_labels=make_regime_labels(),
        benchmark_weights=make_benchmark_weights(),
        config=RegimePortfolioOptimizerConfig(
            max_weight=0.80,
            risk_aversion=0.0,
            cvar_penalty=0.0,
            turnover_penalty=0.0,
            n_candidate_portfolios=100,
            random_state=7,
        ),
    )

    weights = result.weight_table.set_index("regime")

    assert weights.loc[0, "SPY"] >= 0.50
    assert weights.loc[1, "TLT"] >= 0.50
    assert weights.loc[2, "GLD"] >= 0.50


def test_optimizer_rejects_empty_returns() -> None:
    with pytest.raises(RegimePortfolioOptimizerError, match="Asset returns"):
        optimize_regime_portfolios(
            asset_returns=pd.DataFrame(),
            regime_labels=make_regime_labels(),
        )


def test_optimizer_rejects_empty_regimes() -> None:
    with pytest.raises(RegimePortfolioOptimizerError, match="Regime labels"):
        optimize_regime_portfolios(
            asset_returns=make_asset_returns(),
            regime_labels=pd.Series(dtype=int),
        )


def test_optimizer_rejects_bad_benchmark_weights() -> None:
    with pytest.raises(RegimePortfolioOptimizerError, match="sum to 1.0"):
        optimize_regime_portfolios(
            asset_returns=make_asset_returns(),
            regime_labels=make_regime_labels(),
            benchmark_weights={
                "SPY": 0.50,
                "TLT": 0.30,
                "GLD": 0.10,
            },
        )


def test_optimizer_rejects_infeasible_min_weight() -> None:
    with pytest.raises(RegimePortfolioOptimizerError, match="Minimum weight"):
        optimize_regime_portfolios(
            asset_returns=make_asset_returns(),
            regime_labels=make_regime_labels(),
            config=RegimePortfolioOptimizerConfig(
                min_weight=0.50,
                max_weight=0.80,
            ),
        )


def test_optimizer_rejects_infeasible_max_weight() -> None:
    with pytest.raises(RegimePortfolioOptimizerError, match="Maximum weight"):
        optimize_regime_portfolios(
            asset_returns=make_asset_returns(),
            regime_labels=make_regime_labels(),
            config=RegimePortfolioOptimizerConfig(
                min_weight=0.0,
                max_weight=0.20,
            ),
        )
