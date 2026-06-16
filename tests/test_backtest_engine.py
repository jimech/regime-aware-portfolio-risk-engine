import pandas as pd
import pytest

from regime_risk_engine.backtesting.engine import (
    BacktestEngineError,
    BacktestResult,
    build_applied_weight_frame,
    calculate_strategy_gross_returns,
    run_return_backtest,
)


def make_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["SPY"] * 4 + ["TLT"] * 4,
            "return": [
                0.01,
                0.02,
                -0.01,
                0.03,
                0.00,
                0.01,
                0.02,
                -0.01,
            ],
        }
    )


def make_weight_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SPY": 0.60, "TLT": 0.40},
            {"SPY": 0.60, "TLT": 0.40},
            {"SPY": 0.40, "TLT": 0.60},
            {"SPY": 0.50, "TLT": 0.50},
        ],
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )


def test_calculate_strategy_gross_returns_uses_lagged_weights() -> None:
    returns = make_returns()
    weight_frame = make_weight_frame()

    gross_returns = calculate_strategy_gross_returns(
        returns=returns,
        weight_frame=weight_frame,
        weight_lag=1,
    )

    assert list(gross_returns.index) == list(pd.date_range("2020-01-02", periods=3))
    assert gross_returns.name == "gross_return"

    expected_first_return = (0.02 * 0.60) + (0.01 * 0.40)

    assert gross_returns.iloc[0] == pytest.approx(expected_first_return)


def test_calculate_strategy_gross_returns_allows_zero_lag() -> None:
    returns = make_returns()
    weight_frame = make_weight_frame()

    gross_returns = calculate_strategy_gross_returns(
        returns=returns,
        weight_frame=weight_frame,
        weight_lag=0,
    )

    assert list(gross_returns.index) == list(pd.date_range("2020-01-01", periods=4))

    expected_first_return = (0.01 * 0.60) + (0.00 * 0.40)

    assert gross_returns.iloc[0] == pytest.approx(expected_first_return)


def test_build_applied_weight_frame() -> None:
    weight_frame = make_weight_frame()
    return_dates = pd.date_range("2020-01-01", periods=4, freq="D")

    applied_weights = build_applied_weight_frame(
        weight_frame=weight_frame,
        return_dates=return_dates,
        weight_lag=1,
    )

    assert list(applied_weights.index) == list(pd.date_range("2020-01-02", periods=3))
    assert applied_weights.loc[pd.Timestamp("2020-01-02"), "SPY"] == pytest.approx(0.60)
    assert applied_weights.loc[pd.Timestamp("2020-01-04"), "SPY"] == pytest.approx(0.40)


def test_run_return_backtest_returns_result() -> None:
    returns = make_returns()
    weight_frame = make_weight_frame()

    result = run_return_backtest(
        returns=returns,
        weight_frame=weight_frame,
        transaction_cost_bps=10.0,
        weight_lag=1,
    )

    assert isinstance(result, BacktestResult)
    assert len(result.gross_returns) == 3
    assert len(result.net_returns) == 3
    assert len(result.turnover) == 3
    assert len(result.transaction_costs) == 3
    assert len(result.applied_weights) == 3


def test_run_return_backtest_calculates_net_returns_after_costs() -> None:
    returns = make_returns()
    weight_frame = make_weight_frame()

    result = run_return_backtest(
        returns=returns,
        weight_frame=weight_frame,
        transaction_cost_bps=10.0,
        weight_lag=1,
    )

    first_date = pd.Timestamp("2020-01-02")
    second_date = pd.Timestamp("2020-01-03")

    assert result.turnover.loc[first_date] == pytest.approx(0.0)
    assert result.transaction_costs.loc[first_date] == pytest.approx(0.0)
    assert result.net_returns.loc[first_date] == pytest.approx(
        result.gross_returns.loc[first_date]
    )

    assert result.turnover.loc[second_date] == pytest.approx(0.20)
    assert result.transaction_costs.loc[second_date] == pytest.approx(0.0002)
    assert result.net_returns.loc[second_date] == pytest.approx(
        result.gross_returns.loc[second_date] - 0.0002
    )


def test_backtest_rejects_negative_weight_lag() -> None:
    with pytest.raises(BacktestEngineError, match="Weight lag"):
        calculate_strategy_gross_returns(
            returns=make_returns(),
            weight_frame=make_weight_frame(),
            weight_lag=-1,
        )


def test_backtest_rejects_missing_return_columns() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
        }
    )

    with pytest.raises(BacktestEngineError, match="Missing required return"):
        calculate_strategy_gross_returns(
            returns=returns,
            weight_frame=pd.DataFrame(
                [{"SPY": 1.0}],
                index=pd.to_datetime(["2020-01-01"]),
            ),
        )


def test_backtest_rejects_duplicate_return_rows() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "ticker": ["SPY", "SPY"],
            "return": [0.01, 0.02],
        }
    )

    with pytest.raises(BacktestEngineError, match="duplicate"):
        calculate_strategy_gross_returns(
            returns=returns,
            weight_frame=pd.DataFrame(
                [{"SPY": 1.0}],
                index=pd.to_datetime(["2020-01-01"]),
            ),
        )


def test_backtest_rejects_non_datetime_weight_index() -> None:
    weight_frame = pd.DataFrame(
        [{"SPY": 0.60, "TLT": 0.40}],
        index=[0],
    )

    with pytest.raises(BacktestEngineError, match="DatetimeIndex"):
        calculate_strategy_gross_returns(
            returns=make_returns(),
            weight_frame=weight_frame,
        )


def test_backtest_rejects_weight_rows_that_do_not_sum_to_one() -> None:
    weight_frame = pd.DataFrame(
        [{"SPY": 0.60, "TLT": 0.60}],
        index=pd.to_datetime(["2020-01-01"]),
    )

    with pytest.raises(BacktestEngineError, match="sum to 1.0"):
        calculate_strategy_gross_returns(
            returns=make_returns(),
            weight_frame=weight_frame,
        )


def test_backtest_rejects_missing_weight_ticker() -> None:
    weight_frame = pd.DataFrame(
        [{"SPY": 1.0}],
        index=pd.date_range("2020-01-01", periods=1),
    )

    with pytest.raises(BacktestEngineError, match="Missing weight"):
        calculate_strategy_gross_returns(
            returns=make_returns(),
            weight_frame=weight_frame,
        )


def test_backtest_rejects_unknown_weight_ticker() -> None:
    weight_frame = pd.DataFrame(
        [
            {
                "SPY": 0.50,
                "TLT": 0.40,
                "GLD": 0.10,
            }
        ],
        index=pd.date_range("2020-01-01", periods=1),
    )

    with pytest.raises(BacktestEngineError, match="unknown ticker"):
        calculate_strategy_gross_returns(
            returns=make_returns(),
            weight_frame=weight_frame,
        )


def test_backtest_rejects_no_valid_dates_after_lag() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "ticker": ["SPY", "TLT"],
            "return": [0.01, 0.00],
        }
    )
    weight_frame = pd.DataFrame(
        [{"SPY": 0.60, "TLT": 0.40}],
        index=pd.to_datetime(["2020-01-01"]),
    )

    with pytest.raises(BacktestEngineError, match="No valid dates"):
        calculate_strategy_gross_returns(
            returns=returns,
            weight_frame=weight_frame,
            weight_lag=1,
        )
