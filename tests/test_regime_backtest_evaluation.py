import pandas as pd
import pytest

from regime_risk_engine.backtesting.regime_evaluation import (
    RegimeBacktestEvaluation,
    RegimeBacktestEvaluationError,
    build_regime_strategy_return_frame,
    calculate_regime_metric_deltas,
    calculate_regime_strategy_metric_summary,
    evaluate_backtest_by_regime,
)


def make_strategy_returns() -> dict[str, pd.Series]:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")

    return {
        "static": pd.Series(
            [0.01, 0.02, -0.01, -0.02, 0.01, 0.00],
            index=dates,
            name="static",
        ),
        "dynamic": pd.Series(
            [0.02, 0.01, 0.00, -0.01, 0.02, 0.01],
            index=dates,
            name="dynamic",
        ),
    }


def make_regime_labels() -> pd.Series:
    return pd.Series(
        [0, 0, 0, 1, 1, 1],
        index=pd.date_range("2020-01-01", periods=6, freq="D"),
        name="regime",
    )


def test_build_regime_strategy_return_frame() -> None:
    frame = build_regime_strategy_return_frame(
        strategy_returns=make_strategy_returns(),
        regime_labels=make_regime_labels(),
    )

    assert list(frame.columns) == ["static", "dynamic", "regime"]
    assert len(frame) == 6
    assert frame.loc[pd.Timestamp("2020-01-01"), "regime"] == 0
    assert frame.loc[pd.Timestamp("2020-01-04"), "regime"] == 1


def test_calculate_regime_strategy_metric_summary() -> None:
    summary = calculate_regime_strategy_metric_summary(
        strategy_returns=make_strategy_returns(),
        regime_labels=make_regime_labels(),
        confidence_level=0.80,
        annualization_factor=252,
    )

    assert set(summary["regime"]) == {0, 1}
    assert set(summary["strategy"]) == {"static", "dynamic"}
    assert len(summary) == 4
    assert set(summary["observation_count"]) == {3}
    assert "cumulative_return" in summary.columns
    assert "annualized_volatility" in summary.columns
    assert "max_drawdown" in summary.columns
    assert "var" in summary.columns
    assert "cvar" in summary.columns


def test_calculate_regime_metric_deltas() -> None:
    metric_summary = pd.DataFrame(
        {
            "regime": [0, 0, 1, 1],
            "strategy": ["static", "dynamic", "static", "dynamic"],
            "observation_count": [3, 3, 3, 3],
            "sharpe_ratio": [1.0, 1.5, 0.5, 0.25],
            "max_drawdown": [-0.20, -0.10, -0.30, -0.20],
        }
    )

    deltas = calculate_regime_metric_deltas(
        metric_summary=metric_summary,
        benchmark_strategy="static",
        candidate_strategy="dynamic",
    )

    assert set(deltas["regime"]) == {0, 1}
    assert set(deltas["metric"]) == {"sharpe_ratio", "max_drawdown"}

    regime_0_sharpe = deltas[
        (deltas["regime"] == 0) & (deltas["metric"] == "sharpe_ratio")
    ].iloc[0]

    regime_1_drawdown = deltas[
        (deltas["regime"] == 1) & (deltas["metric"] == "max_drawdown")
    ].iloc[0]

    assert regime_0_sharpe["absolute_delta"] == pytest.approx(0.5)
    assert regime_0_sharpe["relative_delta"] == pytest.approx(0.5)
    assert regime_1_drawdown["absolute_delta"] == pytest.approx(0.10)
    assert regime_1_drawdown["relative_delta"] == pytest.approx(1 / 3)


def test_evaluate_backtest_by_regime() -> None:
    evaluation = evaluate_backtest_by_regime(
        strategy_returns=make_strategy_returns(),
        regime_labels=make_regime_labels(),
        confidence_level=0.80,
        annualization_factor=252,
    )

    assert isinstance(evaluation, RegimeBacktestEvaluation)
    assert len(evaluation.regime_return_frame) == 6
    assert len(evaluation.metric_summary) == 4
    assert set(evaluation.metric_deltas["regime"]) == {0, 1}


def test_regime_evaluation_aligns_overlapping_dates() -> None:
    strategy_returns = make_strategy_returns()
    labels = pd.Series(
        [0, 0, 1],
        index=pd.date_range("2020-01-02", periods=3, freq="D"),
        name="regime",
    )

    frame = build_regime_strategy_return_frame(
        strategy_returns=strategy_returns,
        regime_labels=labels,
    )

    assert list(frame.index) == list(pd.date_range("2020-01-02", periods=3))


def test_regime_evaluation_rejects_empty_strategy_mapping() -> None:
    with pytest.raises(RegimeBacktestEvaluationError, match="At least one"):
        build_regime_strategy_return_frame(
            strategy_returns={},
            regime_labels=make_regime_labels(),
        )


def test_regime_evaluation_rejects_non_datetime_strategy_returns() -> None:
    strategy_returns = {
        "static": pd.Series([0.01, 0.02], index=[0, 1]),
    }

    with pytest.raises(RegimeBacktestEvaluationError, match="DatetimeIndex"):
        build_regime_strategy_return_frame(
            strategy_returns=strategy_returns,
            regime_labels=make_regime_labels(),
        )


def test_regime_evaluation_rejects_non_datetime_regime_labels() -> None:
    labels = pd.Series([0, 1], index=[0, 1], name="regime")

    with pytest.raises(RegimeBacktestEvaluationError, match="DatetimeIndex"):
        build_regime_strategy_return_frame(
            strategy_returns=make_strategy_returns(),
            regime_labels=labels,
        )


def test_regime_evaluation_rejects_missing_regime_labels() -> None:
    labels = pd.Series(
        [0, None],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
        name="regime",
    )

    with pytest.raises(RegimeBacktestEvaluationError, match="missing"):
        build_regime_strategy_return_frame(
            strategy_returns=make_strategy_returns(),
            regime_labels=labels,
        )


def test_regime_evaluation_rejects_no_overlapping_dates() -> None:
    labels = pd.Series(
        [0, 1],
        index=pd.to_datetime(["2030-01-01", "2030-01-02"]),
        name="regime",
    )

    with pytest.raises(RegimeBacktestEvaluationError, match="no overlapping dates"):
        build_regime_strategy_return_frame(
            strategy_returns=make_strategy_returns(),
            regime_labels=labels,
        )


def test_regime_metric_deltas_rejects_missing_benchmark() -> None:
    metric_summary = pd.DataFrame(
        {
            "regime": [0],
            "strategy": ["dynamic"],
            "observation_count": [3],
            "sharpe_ratio": [1.0],
        }
    )

    with pytest.raises(RegimeBacktestEvaluationError, match="Benchmark strategy"):
        calculate_regime_metric_deltas(
            metric_summary=metric_summary,
            benchmark_strategy="static",
            candidate_strategy="dynamic",
        )


def test_regime_metric_deltas_rejects_missing_candidate() -> None:
    metric_summary = pd.DataFrame(
        {
            "regime": [0],
            "strategy": ["static"],
            "observation_count": [3],
            "sharpe_ratio": [1.0],
        }
    )

    with pytest.raises(RegimeBacktestEvaluationError, match="Candidate strategy"):
        calculate_regime_metric_deltas(
            metric_summary=metric_summary,
            benchmark_strategy="static",
            candidate_strategy="dynamic",
        )


def test_regime_metric_deltas_rejects_missing_required_columns() -> None:
    metric_summary = pd.DataFrame(
        {
            "strategy": ["static"],
            "sharpe_ratio": [1.0],
        }
    )

    with pytest.raises(RegimeBacktestEvaluationError, match="Missing required"):
        calculate_regime_metric_deltas(metric_summary)
