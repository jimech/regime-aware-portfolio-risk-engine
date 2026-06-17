import pandas as pd
import pytest

from regime_risk_engine.research.regime_optimizer import RegimePortfolioOptimizerConfig
from regime_risk_engine.research.walk_forward_optimizer import (
    WalkForwardRegimeOptimizationResult,
    WalkForwardRegimeOptimizerConfig,
    WalkForwardRegimeOptimizerError,
    run_walk_forward_regime_optimization,
)


def make_asset_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []

    for index, _date in enumerate(dates):
        cycle = index // 20

        if cycle % 3 == 0:
            rows.append({"SPY": 0.004, "TLT": 0.0005, "GLD": 0.0002})
        elif cycle % 3 == 1:
            rows.append({"SPY": -0.006, "TLT": 0.003, "GLD": 0.001})
        else:
            rows.append({"SPY": 0.0005, "TLT": -0.001, "GLD": 0.004})

    return pd.DataFrame(rows, index=dates)


def make_regime_labels() -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    labels = []

    for index, _date in enumerate(dates):
        cycle = index // 20
        labels.append(cycle % 3)

    return pd.Series(labels, index=dates, name="regime")


def make_benchmark_weights() -> dict[str, float]:
    return {
        "SPY": 0.60,
        "TLT": 0.30,
        "GLD": 0.10,
    }


def make_walk_config() -> WalkForwardRegimeOptimizerConfig:
    return WalkForwardRegimeOptimizerConfig(
        train_window=40,
        test_window=10,
        min_training_observations=30,
        annualization_factor=252,
    )


def make_optimizer_config() -> RegimePortfolioOptimizerConfig:
    return RegimePortfolioOptimizerConfig(
        max_weight=0.80,
        risk_aversion=0.25,
        cvar_penalty=0.25,
        turnover_penalty=0.05,
        n_candidate_portfolios=200,
        random_state=7,
    )


def test_run_walk_forward_regime_optimization() -> None:
    result = run_walk_forward_regime_optimization(
        asset_returns=make_asset_returns(),
        regime_labels=make_regime_labels(),
        benchmark_weights=make_benchmark_weights(),
        config=make_walk_config(),
        optimizer_config=make_optimizer_config(),
    )

    assert isinstance(result, WalkForwardRegimeOptimizationResult)
    assert not result.return_frame.empty
    assert not result.dynamic_weight_frame.empty
    assert not result.optimization_diagnostics.empty
    assert not result.metric_summary.empty
    assert not result.metric_deltas.empty
    assert {"static", "dynamic", "regime", "window"}.issubset(
        result.return_frame.columns
    )


def test_walk_forward_weights_sum_to_one() -> None:
    result = run_walk_forward_regime_optimization(
        asset_returns=make_asset_returns(),
        regime_labels=make_regime_labels(),
        benchmark_weights=make_benchmark_weights(),
        config=make_walk_config(),
        optimizer_config=make_optimizer_config(),
    )

    weight_sums = result.dynamic_weight_frame.sum(axis=1)

    assert all(abs(weight_sum - 1.0) < 1e-8 for weight_sum in weight_sums)


def test_walk_forward_starts_after_training_window() -> None:
    result = run_walk_forward_regime_optimization(
        asset_returns=make_asset_returns(),
        regime_labels=make_regime_labels(),
        benchmark_weights=make_benchmark_weights(),
        config=make_walk_config(),
        optimizer_config=make_optimizer_config(),
    )

    expected_first_date = make_asset_returns().index[40]

    assert result.return_frame.index.min() == expected_first_date


def test_walk_forward_rejects_empty_returns() -> None:
    with pytest.raises(WalkForwardRegimeOptimizerError, match="Asset returns"):
        run_walk_forward_regime_optimization(
            asset_returns=pd.DataFrame(),
            regime_labels=make_regime_labels(),
            benchmark_weights=make_benchmark_weights(),
            config=make_walk_config(),
            optimizer_config=make_optimizer_config(),
        )


def test_walk_forward_rejects_empty_regimes() -> None:
    with pytest.raises(WalkForwardRegimeOptimizerError, match="Regime labels"):
        run_walk_forward_regime_optimization(
            asset_returns=make_asset_returns(),
            regime_labels=pd.Series(dtype=int),
            benchmark_weights=make_benchmark_weights(),
            config=make_walk_config(),
            optimizer_config=make_optimizer_config(),
        )


def test_walk_forward_rejects_bad_benchmark_weights() -> None:
    with pytest.raises(WalkForwardRegimeOptimizerError, match="sum to 1.0"):
        run_walk_forward_regime_optimization(
            asset_returns=make_asset_returns(),
            regime_labels=make_regime_labels(),
            benchmark_weights={
                "SPY": 0.50,
                "TLT": 0.30,
                "GLD": 0.10,
            },
            config=make_walk_config(),
            optimizer_config=make_optimizer_config(),
        )


def test_walk_forward_rejects_too_large_training_window() -> None:
    with pytest.raises(WalkForwardRegimeOptimizerError, match="No walk-forward"):
        run_walk_forward_regime_optimization(
            asset_returns=make_asset_returns(),
            regime_labels=make_regime_labels(),
            benchmark_weights=make_benchmark_weights(),
            config=WalkForwardRegimeOptimizerConfig(
                train_window=200,
                test_window=10,
                min_training_observations=30,
            ),
            optimizer_config=make_optimizer_config(),
        )
