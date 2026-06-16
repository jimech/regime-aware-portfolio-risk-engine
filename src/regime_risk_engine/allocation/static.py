from collections.abc import Mapping

import pandas as pd


class StaticAllocationError(ValueError):
    """Raised when static portfolio allocation cannot be calculated."""


def validate_static_weights(
    weights: Mapping[str, float],
    available_tickers: list[str],
) -> dict[str, float]:
    """Validate static portfolio weights against available tickers."""
    if not weights:
        raise StaticAllocationError("Static weights cannot be empty")

    clean_weights = {
        str(ticker).upper().strip(): float(weight) for ticker, weight in weights.items()
    }
    clean_tickers = [str(ticker).upper().strip() for ticker in available_tickers]

    expected_tickers = set(clean_tickers)
    weight_tickers = set(clean_weights)

    missing_tickers = sorted(expected_tickers.difference(weight_tickers))
    extra_tickers = sorted(weight_tickers.difference(expected_tickers))

    if missing_tickers:
        missing = ", ".join(missing_tickers)
        raise StaticAllocationError(
            f"Missing static weight(s) for ticker(s): {missing}"
        )

    if extra_tickers:
        extra = ", ".join(extra_tickers)
        raise StaticAllocationError(
            f"Static weight provided for unknown ticker(s): {extra}"
        )

    if any(weight < 0 for weight in clean_weights.values()):
        raise StaticAllocationError("Static weights must be non-negative")

    weight_sum = sum(clean_weights.values())

    if abs(weight_sum - 1.0) > 1e-8:
        raise StaticAllocationError("Static weights must sum to 1.0")

    return clean_weights


def calculate_static_portfolio_returns(
    returns: pd.DataFrame,
    weights: Mapping[str, float],
    return_col: str = "return",
) -> pd.Series:
    """Calculate static weighted portfolio returns.

    Args:
        returns: Long-format returns with date, ticker, and return columns.
        weights: Static ticker weights.
        return_col: Name of the return column.

    Returns:
        Date-indexed static portfolio return series.
    """
    _validate_returns(returns, return_col=return_col)

    wide_returns = _pivot_returns(returns, return_col=return_col)
    clean_weights = validate_static_weights(
        weights=weights,
        available_tickers=list(wide_returns.columns),
    )

    weight_series = pd.Series(clean_weights, dtype=float)
    portfolio_returns = wide_returns.mul(weight_series, axis=1).sum(axis=1)
    portfolio_returns.name = "static_portfolio_return"

    return portfolio_returns.sort_index()


def build_static_weight_frame(
    weights: Mapping[str, float],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create a date-indexed static weight matrix."""
    if dates.empty:
        raise StaticAllocationError("Dates cannot be empty")

    clean_dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()

    if clean_dates.has_duplicates:
        raise StaticAllocationError("Dates contain duplicate values")

    if not weights:
        raise StaticAllocationError("Static weights cannot be empty")

    clean_weights = {
        str(ticker).upper().strip(): float(weight) for ticker, weight in weights.items()
    }

    if any(weight < 0 for weight in clean_weights.values()):
        raise StaticAllocationError("Static weights must be non-negative")

    weight_sum = sum(clean_weights.values())

    if abs(weight_sum - 1.0) > 1e-8:
        raise StaticAllocationError("Static weights must sum to 1.0")

    weight_frame = pd.DataFrame(
        [clean_weights for _ in clean_dates],
        index=clean_dates,
    )
    weight_frame.index.name = "date"

    return weight_frame.sort_index(axis=1)


def _validate_returns(returns: pd.DataFrame, return_col: str) -> None:
    required_columns = {"date", "ticker", return_col}
    missing_columns = required_columns.difference(returns.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise StaticAllocationError(f"Missing required return column(s): {missing}")

    if returns.empty:
        raise StaticAllocationError("Return data is empty")

    if returns[return_col].isna().any():
        raise StaticAllocationError("Return data contains missing values")

    duplicate_rows = returns.duplicated(subset=["date", "ticker"])

    if duplicate_rows.any():
        raise StaticAllocationError("Return data contains duplicate date/ticker rows")


def _pivot_returns(returns: pd.DataFrame, return_col: str) -> pd.DataFrame:
    normalized_returns = returns[["date", "ticker", return_col]].copy()
    normalized_returns["date"] = pd.to_datetime(normalized_returns["date"])
    normalized_returns["ticker"] = (
        normalized_returns["ticker"].astype(str).str.upper().str.strip()
    )
    normalized_returns[return_col] = pd.to_numeric(
        normalized_returns[return_col],
        errors="coerce",
    )

    wide_returns = normalized_returns.pivot(
        index="date",
        columns="ticker",
        values=return_col,
    ).sort_index()

    if wide_returns.isna().any().any():
        raise StaticAllocationError("Return data contains incomplete ticker coverage")

    return wide_returns
