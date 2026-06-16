import pandas as pd
import pytest

from regime_risk_engine.backtesting.comparison import (
    StrategyComparisonError,
    StrategyComparisonResult,
    build_strategy_metric_summary,
    build_strategy_return_frame,
    calculate_metric_deltas,
    compare_static_and_dynamic_backtests,
)


def make_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")

    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["SPY"] * 5 + ["TLT"] * 5,
            "return": [
                0.01,
                0.02,
                -0.01,
                0.03,
                0.00,
                0.00,
                0.01,
                0.02,
                -0.01,
                0.01,
            ],
        }
    )


def make_static_weight_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SPY": 0.60, "TLT": 0.40},
            {"SPY": 0.60, "TLT": 0.40},
            {"SPY": 0.60, "TLT": 0.40},
            {"SPY": 0.60, "TLT": 0.40},
            {"SPY": 0.60, "TLT": 0.40},
        ],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )


def make_dynamic_weight_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SPY": 0.60, "TLT": 0.40},
            {"SPY": 0.60, "TLT": 0.40},
            {"SPY": 0.40, "TLT": 0.60},
            {"SPY": 0.50, "TLT": 0.50},
            {"SPY": 0.70, "TLT": 0.30},
        ],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )


def test_compare_static_and_dynamic_backtests() -> None:
    result = compare_static_and_dynamic_backtests(
        returns=make_returns(),
        static_weight_frame=make_static_weight_frame(),
        dynamic_weight_frame=make_dynamic_weight_frame(),
        transaction_cost_bps=10.0,
        confidence_level=0.80,
        annualization_factor=252,
    )

    assert isinstance(result, StrategyComparisonResult)
    assert list(result.return_comparison.columns) == ["static", "dynamic"]
    assert list(result.metric_summary.index) == ["static", "dynamic"]
    assert set(result.metric_deltas["metric"]) == set(result.metric_summary.columns)
    assert "cumulative_return" in result.metric_summary.columns
    assert "sharpe_ratio" in result.metric_summary.columns


def test_compare_static_and_dynamic_uses_overlapping_net_return_dates() -> None:
    result = compare_static_and_dynamic_backtests(
        returns=make_returns(),
        static_weight_frame=make_static_weight_frame(),
        dynamic_weight_frame=make_dynamic_weight_frame(),
        transaction_cost_bps=10.0,
        confidence_level=0.80,
    )

    assert list(result.return_comparison.index) == list(
        result.static_backtest.net_returns.index
    )
    assert list(result.return_comparison.index) == list(
        result.dynamic_backtest.net_returns.index
    )


def test_build_strategy_return_frame_aligns_dates() -> None:
    static_returns = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
        name="static",
    )
    dynamic_returns = pd.Series(
        [0.04, 0.05, 0.06],
        index=pd.date_range("2020-01-02", periods=3, freq="D"),
        name="dynamic",
    )

    return_frame = build_strategy_return_frame(
        {
            "static": static_returns,
            "dynamic": dynamic_returns,
        }
    )

    assert list(return_frame.index) == list(pd.date_range("2020-01-02", periods=2))
    assert list(return_frame.columns) == ["static", "dynamic"]


def test_build_strategy_metric_summary() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    static_returns = pd.Series([0.01, -0.01, 0.02, 0.00], index=dates)
    dynamic_returns = pd.Series([0.02, -0.005, 0.01, 0.01], index=dates)

    summary = build_strategy_metric_summary(
        {
            "static": static_returns,
            "dynamic": dynamic_returns,
        },
        confidence_level=0.80,
        annualization_factor=252,
    )

    assert list(summary.index) == ["static", "dynamic"]
    assert "annualized_return" in summary.columns
    assert "max_drawdown" in summary.columns


def test_calculate_metric_deltas() -> None:
    metric_summary = pd.DataFrame(
        {
            "sharpe_ratio": [1.0, 1.5],
            "max_drawdown": [-0.20, -0.10],
        },
        index=["static", "dynamic"],
    )

    deltas = calculate_metric_deltas(
        metric_summary=metric_summary,
        benchmark_strategy="static",
        candidate_strategy="dynamic",
    )

    sharpe_delta = deltas.loc[deltas["metric"] == "sharpe_ratio"].iloc[0]
    drawdown_delta = deltas.loc[deltas["metric"] == "max_drawdown"].iloc[0]

    assert sharpe_delta["absolute_delta"] == pytest.approx(0.5)
    assert sharpe_delta["relative_delta"] == pytest.approx(0.5)
    assert drawdown_delta["absolute_delta"] == pytest.approx(0.10)
    assert drawdown_delta["relative_delta"] == pytest.approx(0.50)


def test_build_strategy_return_frame_rejects_empty_mapping() -> None:
    with pytest.raises(StrategyComparisonError, match="At least one"):
        build_strategy_return_frame({})


def test_build_strategy_return_frame_rejects_non_datetime_index() -> None:
    returns = pd.Series([0.01, 0.02], index=[0, 1])

    with pytest.raises(StrategyComparisonError, match="DatetimeIndex"):
        build_strategy_return_frame({"static": returns})


def test_build_strategy_return_frame_rejects_no_overlapping_dates() -> None:
    static_returns = pd.Series(
        [0.01],
        index=pd.to_datetime(["2020-01-01"]),
    )
    dynamic_returns = pd.Series(
        [0.02],
        index=pd.to_datetime(["2030-01-01"]),
    )

    with pytest.raises(StrategyComparisonError, match="no overlapping dates"):
        build_strategy_return_frame(
            {
                "static": static_returns,
                "dynamic": dynamic_returns,
            }
        )


def test_calculate_metric_deltas_rejects_missing_benchmark() -> None:
    metric_summary = pd.DataFrame(
        {
            "sharpe_ratio": [1.0],
        },
        index=["dynamic"],
    )

    with pytest.raises(StrategyComparisonError, match="Benchmark strategy"):
        calculate_metric_deltas(
            metric_summary=metric_summary,
            benchmark_strategy="static",
            candidate_strategy="dynamic",
        )


def test_calculate_metric_deltas_rejects_missing_candidate() -> None:
    metric_summary = pd.DataFrame(
        {
            "sharpe_ratio": [1.0],
        },
        index=["static"],
    )

    with pytest.raises(StrategyComparisonError, match="Candidate strategy"):
        calculate_metric_deltas(
            metric_summary=metric_summary,
            benchmark_strategy="static",
            candidate_strategy="dynamic",
        )
