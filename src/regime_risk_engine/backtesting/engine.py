from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.allocation.turnover import (
    calculate_net_returns_after_costs,
    calculate_transaction_costs,
    calculate_turnover_series,
)


class BacktestEngineError(ValueError):
    """Raised when strategy backtest returns cannot be calculated."""


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Container for backtest return outputs."""

    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    transaction_costs: pd.Series
    applied_weights: pd.DataFrame


def calculate_strategy_gross_returns(
    returns: pd.DataFrame,
    weight_frame: pd.DataFrame,
    return_col: str = "return",
    weight_lag: int = 1,
) -> pd.Series:
    """Calculate gross strategy returns using lagged portfolio weights.

    Args:
        returns: Long-format returns with date, ticker, and return columns.
        weight_frame: Date-indexed target weights.
        return_col: Name of the return column.
        weight_lag: Number of periods to lag weights before applying returns.

    Returns:
        Date-indexed gross strategy return series.
    """
    if weight_lag < 0:
        raise BacktestEngineError("Weight lag cannot be negative")

    wide_returns = _prepare_wide_returns(returns, return_col=return_col)
    clean_weight_frame = _prepare_weight_frame(weight_frame)

    _validate_matching_tickers(
        return_tickers=list(wide_returns.columns),
        weight_tickers=list(clean_weight_frame.columns),
    )

    applied_weights = clean_weight_frame.shift(weight_lag)

    overlapping_dates = wide_returns.index.intersection(applied_weights.index)

    if overlapping_dates.empty:
        raise BacktestEngineError("Returns and weights have no overlapping dates")

    aligned_returns = wide_returns.loc[overlapping_dates].sort_index()
    aligned_weights = applied_weights.loc[overlapping_dates].sort_index()

    valid_dates = aligned_weights.dropna().index

    if valid_dates.empty:
        raise BacktestEngineError("No valid dates remain after applying the weight lag")

    aligned_returns = aligned_returns.loc[valid_dates]
    aligned_weights = aligned_weights.loc[valid_dates]

    gross_returns = (aligned_returns * aligned_weights).sum(axis=1)
    gross_returns.name = "gross_return"

    return gross_returns


def build_applied_weight_frame(
    weight_frame: pd.DataFrame,
    return_dates: pd.DatetimeIndex,
    weight_lag: int = 1,
) -> pd.DataFrame:
    """Build the weight frame actually applied to strategy returns."""
    if weight_lag < 0:
        raise BacktestEngineError("Weight lag cannot be negative")

    clean_weight_frame = _prepare_weight_frame(weight_frame)
    clean_return_dates = pd.DatetimeIndex(pd.to_datetime(return_dates)).sort_values()

    if clean_return_dates.empty:
        raise BacktestEngineError("Return dates cannot be empty")

    if clean_return_dates.has_duplicates:
        raise BacktestEngineError("Return dates contain duplicate values")

    applied_weights = clean_weight_frame.shift(weight_lag)
    overlapping_dates = clean_return_dates.intersection(applied_weights.index)

    if overlapping_dates.empty:
        raise BacktestEngineError("Return dates and weights have no overlapping dates")

    applied_weights = applied_weights.loc[overlapping_dates].dropna()

    if applied_weights.empty:
        raise BacktestEngineError("No valid dates remain after applying the weight lag")

    return applied_weights


def run_return_backtest(
    returns: pd.DataFrame,
    weight_frame: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    return_col: str = "return",
    weight_lag: int = 1,
) -> BacktestResult:
    """Run a return backtest using a date-indexed weight frame.

    The default one-period weight lag avoids look-ahead bias.
    """
    wide_returns = _prepare_wide_returns(returns, return_col=return_col)
    clean_weight_frame = _prepare_weight_frame(weight_frame)

    _validate_matching_tickers(
        return_tickers=list(wide_returns.columns),
        weight_tickers=list(clean_weight_frame.columns),
    )

    gross_returns = calculate_strategy_gross_returns(
        returns=returns,
        weight_frame=clean_weight_frame,
        return_col=return_col,
        weight_lag=weight_lag,
    )
    applied_weights = build_applied_weight_frame(
        weight_frame=clean_weight_frame,
        return_dates=pd.DatetimeIndex(gross_returns.index),
        weight_lag=weight_lag,
    )

    turnover = calculate_turnover_series(clean_weight_frame)
    turnover = turnover.loc[gross_returns.index]

    transaction_costs = calculate_transaction_costs(
        turnover=turnover,
        transaction_cost_bps=transaction_cost_bps,
    )
    net_returns = calculate_net_returns_after_costs(
        gross_returns=gross_returns,
        transaction_costs=transaction_costs,
    )

    return BacktestResult(
        gross_returns=gross_returns,
        net_returns=net_returns,
        turnover=turnover,
        transaction_costs=transaction_costs,
        applied_weights=applied_weights,
    )


def _prepare_wide_returns(
    returns: pd.DataFrame,
    return_col: str,
) -> pd.DataFrame:
    required_columns = {"date", "ticker", return_col}
    missing_columns = required_columns.difference(returns.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise BacktestEngineError(f"Missing required return column(s): {missing}")

    if returns.empty:
        raise BacktestEngineError("Return data is empty")

    if returns[return_col].isna().any():
        raise BacktestEngineError("Return data contains missing values")

    duplicate_rows = returns.duplicated(subset=["date", "ticker"])

    if duplicate_rows.any():
        raise BacktestEngineError("Return data contains duplicate date/ticker rows")

    normalized_returns = returns[["date", "ticker", return_col]].copy()
    normalized_returns["date"] = pd.to_datetime(normalized_returns["date"])
    normalized_returns["ticker"] = (
        normalized_returns["ticker"].astype(str).str.upper().str.strip()
    )
    normalized_returns[return_col] = pd.to_numeric(
        normalized_returns[return_col],
        errors="coerce",
    )

    if normalized_returns[return_col].isna().any():
        raise BacktestEngineError("Return data contains non-numeric values")

    wide_returns = normalized_returns.pivot(
        index="date",
        columns="ticker",
        values=return_col,
    ).sort_index()

    if wide_returns.isna().any().any():
        raise BacktestEngineError("Return data contains incomplete ticker coverage")

    if wide_returns.empty:
        raise BacktestEngineError("Return data is empty after pivoting")

    return wide_returns


def _prepare_weight_frame(weight_frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(weight_frame.index, pd.DatetimeIndex):
        raise BacktestEngineError("Weight frame index must be a DatetimeIndex")

    if weight_frame.empty:
        raise BacktestEngineError("Weight frame cannot be empty")

    if weight_frame.index.has_duplicates:
        raise BacktestEngineError("Weight frame contains duplicate dates")

    if weight_frame.columns.has_duplicates:
        raise BacktestEngineError("Weight frame contains duplicate tickers")

    if weight_frame.isna().any().any():
        raise BacktestEngineError("Weight frame contains missing values")

    clean_weight_frame = weight_frame.copy()
    clean_weight_frame.columns = [
        str(ticker).upper().strip() for ticker in clean_weight_frame.columns
    ]

    if any(not ticker for ticker in clean_weight_frame.columns):
        raise BacktestEngineError("Weight frame tickers must be non-empty")

    clean_weight_frame = clean_weight_frame.astype(float).sort_index()

    if (clean_weight_frame < 0).any().any():
        raise BacktestEngineError("Weight frame cannot contain negative weights")

    row_sums = clean_weight_frame.sum(axis=1)

    if not row_sums.between(0.99999999, 1.00000001).all():
        raise BacktestEngineError("Each weight row must sum to 1.0")

    return clean_weight_frame


def _validate_matching_tickers(
    return_tickers: list[str],
    weight_tickers: list[str],
) -> None:
    clean_return_tickers = {str(ticker).upper().strip() for ticker in return_tickers}
    clean_weight_tickers = {str(ticker).upper().strip() for ticker in weight_tickers}

    missing_weights = sorted(clean_return_tickers.difference(clean_weight_tickers))
    extra_weights = sorted(clean_weight_tickers.difference(clean_return_tickers))

    if missing_weights:
        missing = ", ".join(missing_weights)
        raise BacktestEngineError(f"Missing weight column(s) for ticker(s): {missing}")

    if extra_weights:
        extra = ", ".join(extra_weights)
        raise BacktestEngineError(
            f"Weight column provided for unknown ticker(s): {extra}"
        )
