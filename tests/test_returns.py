import numpy as np
import pandas as pd
import pytest

from regime_risk_engine.data.returns import (
    RETURN_COLUMNS,
    RETURN_INDEX_COLUMNS,
    ReturnCalculationError,
    build_processed_returns_dataset,
    calculate_return_index,
    calculate_returns,
)


def make_price_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                ]
            ),
            "ticker": ["SPY", "SPY", "SPY", "TLT", "TLT", "TLT"],
            "adj_close": [100.0, 110.0, 121.0, 200.0, 190.0, 209.0],
        }
    )


def test_calculate_simple_returns() -> None:
    prices = make_price_data()

    returns = calculate_returns(prices, return_type="simple")

    assert list(returns.columns) == RETURN_COLUMNS
    assert len(returns) == 4

    spy_returns = returns[returns["ticker"] == "SPY"]["return"].tolist()
    tlt_returns = returns[returns["ticker"] == "TLT"]["return"].tolist()

    assert spy_returns == pytest.approx([0.10, 0.10])
    assert tlt_returns == pytest.approx([-0.05, 0.10])


def test_calculate_log_returns() -> None:
    prices = make_price_data()

    returns = calculate_returns(prices, return_type="log")

    spy_returns = returns[returns["ticker"] == "SPY"]["return"].tolist()

    assert spy_returns == pytest.approx([np.log(1.10), np.log(1.10)])


def test_calculate_returns_drops_first_row_per_ticker() -> None:
    prices = make_price_data()

    returns = calculate_returns(prices)

    assert returns.groupby("ticker").size().to_dict() == {
        "SPY": 2,
        "TLT": 2,
    }


def test_calculate_returns_rejects_missing_columns() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
        }
    )

    with pytest.raises(ReturnCalculationError, match="Missing required price column"):
        calculate_returns(prices)


def test_calculate_returns_rejects_empty_data() -> None:
    prices = pd.DataFrame(columns=["date", "ticker", "adj_close"])

    with pytest.raises(ReturnCalculationError, match="Price data is empty"):
        calculate_returns(prices)


def test_calculate_returns_rejects_duplicate_date_ticker_rows() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "ticker": ["SPY", "SPY"],
            "adj_close": [100.0, 101.0],
        }
    )

    with pytest.raises(ReturnCalculationError, match="duplicate"):
        calculate_returns(prices)


def test_calculate_returns_rejects_non_positive_prices() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "ticker": ["SPY", "SPY"],
            "adj_close": [100.0, 0.0],
        }
    )

    with pytest.raises(ReturnCalculationError, match="non-positive"):
        calculate_returns(prices)


def test_calculate_return_index() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-02", "2020-01-03"]
            ),
            "ticker": ["SPY", "SPY", "TLT", "TLT"],
            "return": [0.10, 0.10, -0.05, 0.10],
        }
    )

    return_index = calculate_return_index(returns, base_value=1.0)

    assert list(return_index.columns) == RETURN_INDEX_COLUMNS

    spy_index = return_index[return_index["ticker"] == "SPY"]["return_index"].tolist()

    assert spy_index == pytest.approx([1.10, 1.21])


def test_calculate_return_index_rejects_empty_data() -> None:
    returns = pd.DataFrame(columns=["date", "ticker", "return"])

    with pytest.raises(ReturnCalculationError, match="Return data is empty"):
        calculate_return_index(returns)


def test_build_processed_returns_dataset() -> None:
    prices = make_price_data()

    processed = build_processed_returns_dataset(prices)

    assert set(processed.columns) == {"date", "ticker", "return", "return_index"}
    assert len(processed) == 4
    assert processed["return_index"].notna().all()


def test_build_processed_returns_dataset_with_log_returns() -> None:
    prices = make_price_data()

    processed = build_processed_returns_dataset(prices, return_type="log")

    assert set(processed.columns) == {"date", "ticker", "return"}
    assert "return_index" not in processed.columns
