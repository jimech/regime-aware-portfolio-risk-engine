import math

import pandas as pd
import pytest

from regime_risk_engine.backtesting.diagnostics import (
    BacktestDiagnostics,
    BacktestDiagnosticsError,
    build_strategy_diagnostics,
    build_strategy_return_frame,
    calculate_cumulative_return_series,
    calculate_drawdown_series,
    calculate_rolling_sharpe_ratio,
    calculate_rolling_volatility,
)


def make_returns() -> pd.Series:
    return pd.Series(
        [0.10, -0.05, 0.02, -0.10, 0.04],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
        name="strategy",
    )


def test_calculate_cumulative_return_series() -> None:
    returns = make_returns()

    cumulative_returns = calculate_cumulative_return_series(returns)

    assert cumulative_returns.name == "cumulative_return"
    assert cumulative_returns.iloc[0] == pytest.approx(0.10)
    assert cumulative_returns.iloc[1] == pytest.approx((1.10 * 0.95) - 1.0)


def test_calculate_drawdown_series() -> None:
    returns = make_returns()

    drawdowns = calculate_drawdown_series(returns)

    assert drawdowns.name == "drawdown"
    assert drawdowns.iloc[0] == pytest.approx(0.0)
    assert drawdowns.iloc[1] == pytest.approx(-0.05)


def test_calculate_rolling_volatility() -> None:
    returns = make_returns()

    rolling_volatility = calculate_rolling_volatility(
        returns=returns,
        window=3,
        annualization_factor=252,
    )

    expected = returns.iloc[:3].std(ddof=1) * math.sqrt(252)

    assert rolling_volatility.name == "rolling_volatility"
    assert pd.isna(rolling_volatility.iloc[0])
    assert pd.isna(rolling_volatility.iloc[1])
    assert rolling_volatility.iloc[2] == pytest.approx(expected)


def test_calculate_rolling_sharpe_ratio() -> None:
    returns = make_returns()

    rolling_sharpe = calculate_rolling_sharpe_ratio(
        returns=returns,
        window=3,
        risk_free_rate=0.0,
        annualization_factor=252,
    )

    expected_return = returns.iloc[:3].mean() * 252
    expected_volatility = returns.iloc[:3].std(ddof=1) * math.sqrt(252)
    expected_sharpe = expected_return / expected_volatility

    assert rolling_sharpe.name == "rolling_sharpe"
    assert pd.isna(rolling_sharpe.iloc[0])
    assert pd.isna(rolling_sharpe.iloc[1])
    assert rolling_sharpe.iloc[2] == pytest.approx(expected_sharpe)


def test_build_strategy_return_frame_aligns_dates() -> None:
    static_returns = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    dynamic_returns = pd.Series(
        [0.04, 0.05, 0.06],
        index=pd.date_range("2020-01-02", periods=3, freq="D"),
    )

    return_frame = build_strategy_return_frame(
        {
            "static": static_returns,
            "dynamic": dynamic_returns,
        }
    )

    assert list(return_frame.columns) == ["static", "dynamic"]
    assert list(return_frame.index) == list(pd.date_range("2020-01-02", periods=2))


def test_build_strategy_diagnostics() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    static_returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00], index=dates)
    dynamic_returns = pd.Series([0.02, 0.01, 0.00, 0.02, -0.01], index=dates)

    diagnostics = build_strategy_diagnostics(
        {
            "static": static_returns,
            "dynamic": dynamic_returns,
        },
        rolling_window=3,
        annualization_factor=252,
    )

    assert isinstance(diagnostics, BacktestDiagnostics)
    assert list(diagnostics.cumulative_returns.columns) == ["static", "dynamic"]
    assert list(diagnostics.drawdowns.columns) == ["static", "dynamic"]
    assert list(diagnostics.rolling_volatility.columns) == ["static", "dynamic"]
    assert list(diagnostics.rolling_sharpe.columns) == ["static", "dynamic"]


def test_diagnostics_reject_empty_return_series() -> None:
    returns = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    with pytest.raises(BacktestDiagnosticsError, match="cannot be empty"):
        calculate_cumulative_return_series(returns)


def test_diagnostics_reject_non_datetime_index() -> None:
    returns = pd.Series([0.01, 0.02], index=[0, 1])

    with pytest.raises(BacktestDiagnosticsError, match="DatetimeIndex"):
        calculate_cumulative_return_series(returns)


def test_diagnostics_reject_missing_returns() -> None:
    returns = pd.Series(
        [0.01, None],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    with pytest.raises(BacktestDiagnosticsError, match="missing values"):
        calculate_cumulative_return_series(returns)


def test_diagnostics_reject_invalid_window() -> None:
    returns = make_returns()

    with pytest.raises(BacktestDiagnosticsError, match="Rolling window"):
        calculate_rolling_volatility(
            returns=returns,
            window=1,
        )


def test_diagnostics_reject_invalid_annualization_factor() -> None:
    returns = make_returns()

    with pytest.raises(BacktestDiagnosticsError, match="Annualization factor"):
        calculate_rolling_volatility(
            returns=returns,
            window=3,
            annualization_factor=0,
        )


def test_diagnostics_reject_empty_strategy_mapping() -> None:
    with pytest.raises(BacktestDiagnosticsError, match="At least one"):
        build_strategy_return_frame({})


def test_diagnostics_reject_no_overlapping_dates() -> None:
    static_returns = pd.Series(
        [0.01],
        index=pd.to_datetime(["2020-01-01"]),
    )
    dynamic_returns = pd.Series(
        [0.02],
        index=pd.to_datetime(["2030-01-01"]),
    )

    with pytest.raises(BacktestDiagnosticsError, match="no overlapping dates"):
        build_strategy_return_frame(
            {
                "static": static_returns,
                "dynamic": dynamic_returns,
            }
        )


def test_rolling_sharpe_rejects_invalid_risk_free_rate() -> None:
    returns = make_returns()

    with pytest.raises(BacktestDiagnosticsError, match="Risk-free rate"):
        calculate_rolling_sharpe_ratio(
            returns=returns,
            window=3,
            risk_free_rate=-1.0,
        )
