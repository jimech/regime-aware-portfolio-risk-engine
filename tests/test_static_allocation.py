import pandas as pd
import pytest

from regime_risk_engine.allocation.static import (
    StaticAllocationError,
    build_static_weight_frame,
    calculate_static_portfolio_returns,
    validate_static_weights,
)


def make_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")

    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["SPY"] * 3 + ["TLT"] * 3,
            "return": [0.01, 0.02, -0.01, 0.00, 0.01, 0.02],
        }
    )


def make_weights() -> dict[str, float]:
    return {
        "SPY": 0.6,
        "TLT": 0.4,
    }


def test_validate_static_weights_success() -> None:
    weights = validate_static_weights(
        weights=make_weights(),
        available_tickers=["SPY", "TLT"],
    )

    assert weights == {
        "SPY": 0.6,
        "TLT": 0.4,
    }


def test_calculate_static_portfolio_returns() -> None:
    returns = make_returns()

    portfolio_returns = calculate_static_portfolio_returns(
        returns=returns,
        weights=make_weights(),
    )

    assert portfolio_returns.name == "static_portfolio_return"
    assert len(portfolio_returns) == 3
    assert portfolio_returns.iloc[0] == pytest.approx(0.006)
    assert portfolio_returns.iloc[1] == pytest.approx(0.016)
    assert portfolio_returns.iloc[2] == pytest.approx(0.002)


def test_build_static_weight_frame() -> None:
    dates = pd.date_range("2020-01-01", periods=3, freq="D")

    weight_frame = build_static_weight_frame(
        weights=make_weights(),
        dates=dates,
    )

    assert list(weight_frame.columns) == ["SPY", "TLT"]
    assert len(weight_frame) == 3
    assert weight_frame.loc[pd.Timestamp("2020-01-01"), "SPY"] == pytest.approx(0.6)
    assert weight_frame.loc[pd.Timestamp("2020-01-03"), "TLT"] == pytest.approx(0.4)


def test_validate_static_weights_rejects_empty_weights() -> None:
    with pytest.raises(StaticAllocationError, match="empty"):
        validate_static_weights({}, available_tickers=["SPY", "TLT"])


def test_validate_static_weights_rejects_missing_ticker_weight() -> None:
    with pytest.raises(StaticAllocationError, match="Missing static weight"):
        validate_static_weights(
            weights={"SPY": 1.0},
            available_tickers=["SPY", "TLT"],
        )


def test_validate_static_weights_rejects_unknown_ticker_weight() -> None:
    with pytest.raises(StaticAllocationError, match="unknown ticker"):
        validate_static_weights(
            weights={
                "SPY": 0.5,
                "TLT": 0.4,
                "GLD": 0.1,
            },
            available_tickers=["SPY", "TLT"],
        )


def test_validate_static_weights_rejects_negative_weight() -> None:
    with pytest.raises(StaticAllocationError, match="non-negative"):
        validate_static_weights(
            weights={
                "SPY": 1.1,
                "TLT": -0.1,
            },
            available_tickers=["SPY", "TLT"],
        )


def test_validate_static_weights_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(StaticAllocationError, match="sum to 1.0"):
        validate_static_weights(
            weights={
                "SPY": 0.7,
                "TLT": 0.7,
            },
            available_tickers=["SPY", "TLT"],
        )


def test_calculate_static_portfolio_returns_rejects_missing_columns() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
        }
    )

    with pytest.raises(StaticAllocationError, match="Missing required return"):
        calculate_static_portfolio_returns(
            returns=returns,
            weights={"SPY": 1.0},
        )


def test_calculate_static_portfolio_returns_rejects_duplicate_rows() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "ticker": ["SPY", "SPY"],
            "return": [0.01, 0.02],
        }
    )

    with pytest.raises(StaticAllocationError, match="duplicate"):
        calculate_static_portfolio_returns(
            returns=returns,
            weights={"SPY": 1.0},
        )


def test_calculate_static_portfolio_returns_rejects_incomplete_coverage() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
            "ticker": ["SPY", "SPY", "TLT"],
            "return": [0.01, 0.02, 0.00],
        }
    )

    with pytest.raises(StaticAllocationError, match="incomplete ticker coverage"):
        calculate_static_portfolio_returns(
            returns=returns,
            weights=make_weights(),
        )


def test_build_static_weight_frame_rejects_empty_dates() -> None:
    dates = pd.DatetimeIndex([])

    with pytest.raises(StaticAllocationError, match="Dates cannot be empty"):
        build_static_weight_frame(
            weights=make_weights(),
            dates=dates,
        )
