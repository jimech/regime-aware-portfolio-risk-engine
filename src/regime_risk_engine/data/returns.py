from typing import Literal

import numpy as np
import pandas as pd

ReturnType = Literal["simple", "log"]

RETURN_COLUMNS = ["date", "ticker", "return"]
RETURN_INDEX_COLUMNS = ["date", "ticker", "return_index"]


class ReturnCalculationError(ValueError):
    """Raised when returns cannot be calculated from price data."""


def calculate_returns(
    prices: pd.DataFrame,
    return_type: ReturnType = "simple",
    price_col: str = "adj_close",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Calculate daily returns from normalized price data.

    Args:
        prices: Long-format price data.
        return_type: Type of return to calculate: simple or log.
        price_col: Name of the price column.
        date_col: Name of the date column.
        ticker_col: Name of the ticker column.

    Returns:
        Long-format returns with date, ticker, and return columns.

    Raises:
        ReturnCalculationError: If input data is invalid.
    """
    _validate_price_data_for_returns(prices, price_col, date_col, ticker_col)

    sorted_prices = prices[[date_col, ticker_col, price_col]].copy()
    sorted_prices[date_col] = pd.to_datetime(sorted_prices[date_col])
    sorted_prices[ticker_col] = sorted_prices[ticker_col].astype(str).str.upper()
    sorted_prices = sorted_prices.sort_values([ticker_col, date_col])

    previous_prices = sorted_prices.groupby(ticker_col)[price_col].shift(1)

    if return_type == "simple":
        sorted_prices["return"] = sorted_prices[price_col] / previous_prices - 1.0
    elif return_type == "log":
        sorted_prices["return"] = np.log(sorted_prices[price_col] / previous_prices)
    else:
        raise ReturnCalculationError(f"Unsupported return type: {return_type}")

    returns = sorted_prices.rename(
        columns={
            date_col: "date",
            ticker_col: "ticker",
        }
    )[RETURN_COLUMNS]

    return returns.dropna(subset=["return"]).reset_index(drop=True)


def calculate_return_index(
    returns: pd.DataFrame,
    return_col: str = "return",
    date_col: str = "date",
    ticker_col: str = "ticker",
    base_value: float = 1.0,
) -> pd.DataFrame:
    """Calculate cumulative return index from simple returns.

    Args:
        returns: Long-format return data.
        return_col: Name of the simple return column.
        date_col: Name of the date column.
        ticker_col: Name of the ticker column.
        base_value: Starting value for the return index.

    Returns:
        Long-format return index with date, ticker, and return_index columns.

    Raises:
        ReturnCalculationError: If input data is invalid.
    """
    required_columns = {date_col, ticker_col, return_col}
    missing_columns = required_columns.difference(returns.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ReturnCalculationError(f"Missing required return column(s): {missing}")

    if returns.empty:
        raise ReturnCalculationError("Return data is empty")

    if base_value <= 0:
        raise ReturnCalculationError("Base value must be positive")

    sorted_returns = returns[[date_col, ticker_col, return_col]].copy()
    sorted_returns[date_col] = pd.to_datetime(sorted_returns[date_col])
    sorted_returns[ticker_col] = sorted_returns[ticker_col].astype(str).str.upper()
    sorted_returns = sorted_returns.sort_values([ticker_col, date_col])

    gross_returns = 1.0 + sorted_returns[return_col]
    sorted_returns["return_index"] = (
        base_value * gross_returns.groupby(sorted_returns[ticker_col]).cumprod()
    )

    return_index = sorted_returns.rename(
        columns={
            date_col: "date",
            ticker_col: "ticker",
        }
    )[RETURN_INDEX_COLUMNS]

    return return_index.reset_index(drop=True)


def build_processed_returns_dataset(
    prices: pd.DataFrame,
    return_type: ReturnType = "simple",
) -> pd.DataFrame:
    """Build processed return dataset from normalized price data.

    Args:
        prices: Long-format price data.
        return_type: Type of return to calculate.

    Returns:
        DataFrame with date, ticker, return, and return_index columns.

    Raises:
        ReturnCalculationError: If log returns are requested with return index.
    """
    returns = calculate_returns(prices, return_type=return_type)

    if return_type == "log":
        return returns

    return_index = calculate_return_index(returns)

    return returns.merge(return_index, on=["date", "ticker"], how="left")


def _validate_price_data_for_returns(
    prices: pd.DataFrame,
    price_col: str,
    date_col: str,
    ticker_col: str,
) -> None:
    required_columns = {date_col, ticker_col, price_col}
    missing_columns = required_columns.difference(prices.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ReturnCalculationError(f"Missing required price column(s): {missing}")

    if prices.empty:
        raise ReturnCalculationError("Price data is empty")

    duplicate_rows = prices.duplicated(subset=[date_col, ticker_col])

    if duplicate_rows.any():
        raise ReturnCalculationError("Price data contains duplicate date/ticker rows")

    missing_prices = prices[price_col].isna()

    if missing_prices.any():
        raise ReturnCalculationError("Price data contains missing prices")

    non_positive_prices = prices[price_col] <= 0

    if non_positive_prices.any():
        raise ReturnCalculationError("Price data contains non-positive prices")
