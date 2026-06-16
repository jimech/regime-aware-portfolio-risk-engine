import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.risk.metrics import DEFAULT_ANNUALIZATION_FACTOR


class BacktestDiagnosticsError(ValueError):
    """Raised when backtest diagnostics cannot be calculated."""


@dataclass(frozen=True, slots=True)
class BacktestDiagnostics:
    """Container for strategy backtest diagnostics."""

    cumulative_returns: pd.DataFrame
    drawdowns: pd.DataFrame
    rolling_volatility: pd.DataFrame
    rolling_sharpe: pd.DataFrame


def calculate_cumulative_return_series(returns: pd.Series) -> pd.Series:
    """Calculate cumulative simple return series."""
    clean_returns = _validate_return_series(returns)

    cumulative_returns = (1.0 + clean_returns).cumprod() - 1.0

    return pd.Series(
        cumulative_returns,
        index=clean_returns.index,
        name="cumulative_return",
    )


def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate drawdown series from simple returns."""
    clean_returns = _validate_return_series(returns)

    return_index = (1.0 + clean_returns).cumprod()
    running_peak = return_index.cummax()
    drawdowns = return_index / running_peak - 1.0

    return pd.Series(
        drawdowns,
        index=clean_returns.index,
        name="drawdown",
    )


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> pd.Series:
    """Calculate rolling annualized volatility."""
    clean_returns = _validate_return_series(returns)
    _validate_window(window)
    _validate_annualization_factor(annualization_factor)

    rolling_volatility = clean_returns.rolling(window=window, min_periods=window).std(
        ddof=1
    ) * math.sqrt(annualization_factor)

    return pd.Series(
        rolling_volatility,
        index=clean_returns.index,
        name="rolling_volatility",
    )


def calculate_rolling_sharpe_ratio(
    returns: pd.Series,
    window: int,
    risk_free_rate: float = 0.0,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> pd.Series:
    """Calculate rolling annualized Sharpe ratio."""
    clean_returns = _validate_return_series(returns)
    _validate_window(window)
    _validate_annualization_factor(annualization_factor)
    _validate_risk_free_rate(risk_free_rate)

    periodic_risk_free_rate = (
        (1.0 + risk_free_rate) ** (1.0 / annualization_factor)
    ) - 1.0

    excess_returns = clean_returns - periodic_risk_free_rate
    rolling_excess_return = (
        excess_returns.rolling(window=window, min_periods=window).mean()
        * annualization_factor
    )
    rolling_volatility = calculate_rolling_volatility(
        returns=clean_returns,
        window=window,
        annualization_factor=annualization_factor,
    )

    rolling_sharpe = rolling_excess_return / rolling_volatility
    rolling_sharpe = rolling_sharpe.replace([float("inf"), float("-inf")], pd.NA)

    return pd.Series(
        rolling_sharpe,
        index=clean_returns.index,
        name="rolling_sharpe",
    )


def build_strategy_diagnostics(
    strategy_returns: Mapping[str, pd.Series],
    rolling_window: int,
    risk_free_rate: float = 0.0,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> BacktestDiagnostics:
    """Build diagnostics for one or more strategy return series."""
    return_frame = build_strategy_return_frame(strategy_returns)

    cumulative_returns = _calculate_metric_frame(
        return_frame=return_frame,
        metric_function=calculate_cumulative_return_series,
    )
    drawdowns = _calculate_metric_frame(
        return_frame=return_frame,
        metric_function=calculate_drawdown_series,
    )
    rolling_volatility = _calculate_rolling_volatility_frame(
        return_frame=return_frame,
        rolling_window=rolling_window,
        annualization_factor=annualization_factor,
    )
    rolling_sharpe = _calculate_rolling_sharpe_frame(
        return_frame=return_frame,
        rolling_window=rolling_window,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
    )

    return BacktestDiagnostics(
        cumulative_returns=cumulative_returns,
        drawdowns=drawdowns,
        rolling_volatility=rolling_volatility,
        rolling_sharpe=rolling_sharpe,
    )


def build_strategy_return_frame(
    strategy_returns: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Align strategy return series on overlapping dates."""
    if not strategy_returns:
        raise BacktestDiagnosticsError(
            "At least one strategy return series is required"
        )

    cleaned_returns: dict[str, pd.Series] = {}

    for strategy_name, returns in strategy_returns.items():
        clean_name = str(strategy_name).strip()

        if not clean_name:
            raise BacktestDiagnosticsError("Strategy names must be non-empty")

        if clean_name in cleaned_returns:
            raise BacktestDiagnosticsError(f"Duplicate strategy name: {clean_name}")

        cleaned_returns[clean_name] = _validate_strategy_returns(
            strategy_name=clean_name,
            returns=returns,
        )

    return_frame = pd.concat(cleaned_returns.values(), axis=1, join="inner")

    if return_frame.empty:
        raise BacktestDiagnosticsError("Strategy returns have no overlapping dates")

    return_frame.index.name = "date"

    return return_frame.sort_index()


def _calculate_metric_frame(
    return_frame: pd.DataFrame,
    metric_function: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}

    for strategy_name in return_frame.columns:
        strategy_returns = return_frame[str(strategy_name)]
        result = metric_function(strategy_returns)
        rows[str(strategy_name)] = pd.Series(
            result,
            index=strategy_returns.index,
            name=strategy_name,
        )

    metric_frame = pd.concat(rows.values(), axis=1)
    metric_frame.index.name = "date"

    return metric_frame


def _calculate_rolling_volatility_frame(
    return_frame: pd.DataFrame,
    rolling_window: int,
    annualization_factor: int,
) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}

    for strategy_name in return_frame.columns:
        strategy_returns = return_frame[str(strategy_name)]
        result = calculate_rolling_volatility(
            returns=strategy_returns,
            window=rolling_window,
            annualization_factor=annualization_factor,
        )
        rows[str(strategy_name)] = pd.Series(
            result,
            index=strategy_returns.index,
            name=strategy_name,
        )

    volatility_frame = pd.concat(rows.values(), axis=1)
    volatility_frame.index.name = "date"

    return volatility_frame


def _calculate_rolling_sharpe_frame(
    return_frame: pd.DataFrame,
    rolling_window: int,
    risk_free_rate: float,
    annualization_factor: int,
) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}

    for strategy_name in return_frame.columns:
        strategy_returns = return_frame[str(strategy_name)]
        result = calculate_rolling_sharpe_ratio(
            returns=strategy_returns,
            window=rolling_window,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
        )
        rows[str(strategy_name)] = pd.Series(
            result,
            index=strategy_returns.index,
            name=strategy_name,
        )

    sharpe_frame = pd.concat(rows.values(), axis=1)
    sharpe_frame.index.name = "date"

    return sharpe_frame


def _validate_return_series(returns: pd.Series) -> pd.Series:
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise BacktestDiagnosticsError("Return series index must be a DatetimeIndex")

    if returns.empty:
        raise BacktestDiagnosticsError("Return series cannot be empty")

    if returns.index.has_duplicates:
        raise BacktestDiagnosticsError("Return series contains duplicate dates")

    clean_returns = pd.to_numeric(returns, errors="coerce")

    if clean_returns.isna().any():
        raise BacktestDiagnosticsError("Return series contains missing values")

    return pd.Series(
        clean_returns,
        index=returns.index,
        name=returns.name,
    ).sort_index()


def _validate_strategy_returns(
    strategy_name: str,
    returns: pd.Series,
) -> pd.Series:
    try:
        clean_returns = _validate_return_series(returns)
    except BacktestDiagnosticsError as error:
        raise BacktestDiagnosticsError(
            f"Invalid return series for strategy {strategy_name}: {error}"
        ) from error

    clean_returns.name = strategy_name

    return clean_returns


def _validate_window(window: int) -> None:
    if window <= 1:
        raise BacktestDiagnosticsError("Rolling window must be greater than 1")


def _validate_annualization_factor(annualization_factor: int) -> None:
    if annualization_factor <= 0:
        raise BacktestDiagnosticsError("Annualization factor must be positive")


def _validate_risk_free_rate(risk_free_rate: float) -> None:
    if risk_free_rate <= -1.0:
        raise BacktestDiagnosticsError("Risk-free rate must be greater than -100%")
