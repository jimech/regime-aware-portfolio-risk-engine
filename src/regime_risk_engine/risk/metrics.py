import math
from collections.abc import Mapping

import pandas as pd


class RiskMetricError(ValueError):
    """Raised when risk metrics cannot be calculated."""


DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_ANNUALIZATION_FACTOR = 252


def cumulative_return(returns: pd.Series) -> float:
    """Calculate cumulative simple return."""
    clean_returns = _validate_return_series(returns)

    result = (1.0 + clean_returns).prod() - 1.0

    return float(result)


def annualized_return(
    returns: pd.Series,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    """Calculate annualized simple return."""
    clean_returns = _validate_return_series(returns)
    _validate_annualization_factor(annualization_factor)

    periods = len(clean_returns)
    total_return = cumulative_return(clean_returns)

    result = (1.0 + total_return) ** (annualization_factor / periods) - 1.0

    return float(result)


def annualized_volatility(
    returns: pd.Series,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    """Calculate annualized volatility."""
    clean_returns = _validate_return_series(returns)
    _validate_annualization_factor(annualization_factor)

    if len(clean_returns) < 2:
        return float("nan")

    result = clean_returns.std(ddof=1) * math.sqrt(annualization_factor)

    return float(result)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    """Calculate annualized Sharpe ratio.

    Args:
        returns: Periodic simple returns.
        risk_free_rate: Annual risk-free rate.
        annualization_factor: Number of periods per year.
    """
    portfolio_return = annualized_return(
        returns,
        annualization_factor=annualization_factor,
    )
    portfolio_volatility = annualized_volatility(
        returns,
        annualization_factor=annualization_factor,
    )

    if portfolio_volatility == 0 or math.isnan(portfolio_volatility):
        return float("nan")

    return float((portfolio_return - risk_free_rate) / portfolio_volatility)


def sortino_ratio(
    returns: pd.Series,
    target_return: float = 0.0,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    """Calculate annualized Sortino ratio.

    Args:
        returns: Periodic simple returns.
        target_return: Periodic minimum acceptable return.
        annualization_factor: Number of periods per year.
    """
    clean_returns = _validate_return_series(returns)
    _validate_annualization_factor(annualization_factor)

    downside_returns = clean_returns[clean_returns < target_return] - target_return

    if len(downside_returns) < 2:
        return float("nan")

    downside_deviation = downside_returns.std(ddof=1) * math.sqrt(annualization_factor)

    if downside_deviation == 0 or math.isnan(float(downside_deviation)):
        return float("nan")

    annual_target_return = (1.0 + target_return) ** annualization_factor - 1.0
    excess_return = (
        annualized_return(
            clean_returns,
            annualization_factor=annualization_factor,
        )
        - annual_target_return
    )

    return float(excess_return / downside_deviation)


def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from simple returns."""
    clean_returns = _validate_return_series(returns)

    return_index = (1.0 + clean_returns).cumprod()
    running_peak = return_index.cummax()
    drawdowns = return_index / running_peak - 1.0

    return float(drawdowns.min())


def historical_var(
    returns: pd.Series,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> float:
    """Calculate historical Value at Risk as a positive loss value."""
    clean_returns = _validate_return_series(returns)
    _validate_confidence_level(confidence_level)

    quantile = 1.0 - confidence_level
    value_at_risk = -clean_returns.quantile(quantile)

    return float(value_at_risk)


def historical_cvar(
    returns: pd.Series,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> float:
    """Calculate historical Conditional Value at Risk as a positive loss value."""
    clean_returns = _validate_return_series(returns)
    _validate_confidence_level(confidence_level)

    quantile = 1.0 - confidence_level
    threshold = clean_returns.quantile(quantile)
    tail_returns = clean_returns[clean_returns <= threshold]

    if tail_returns.empty:
        return float("nan")

    expected_shortfall = -tail_returns.mean()

    return float(expected_shortfall)


def summarize_risk_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> dict[str, float]:
    """Calculate a standard risk metric summary for one return series."""
    return {
        "cumulative_return": cumulative_return(returns),
        "annualized_return": annualized_return(
            returns,
            annualization_factor=annualization_factor,
        ),
        "annualized_volatility": annualized_volatility(
            returns,
            annualization_factor=annualization_factor,
        ),
        "sharpe_ratio": sharpe_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
        ),
        "sortino_ratio": sortino_ratio(
            returns,
            target_return=0.0,
            annualization_factor=annualization_factor,
        ),
        "max_drawdown": max_drawdown(returns),
        "var": historical_var(
            returns,
            confidence_level=confidence_level,
        ),
        "cvar": historical_cvar(
            returns,
            confidence_level=confidence_level,
        ),
    }


def summarize_multiple_return_series(
    returns_by_name: Mapping[str, pd.Series],
    risk_free_rate: float = 0.0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
) -> pd.DataFrame:
    """Calculate risk metric summaries for multiple return series."""
    if not returns_by_name:
        raise RiskMetricError("At least one return series is required")

    rows: list[dict[str, float | str]] = []

    for name, returns in returns_by_name.items():
        summary = summarize_risk_metrics(
            returns=returns,
            risk_free_rate=risk_free_rate,
            confidence_level=confidence_level,
            annualization_factor=annualization_factor,
        )
        rows.append({"name": name, **summary})

    return pd.DataFrame(rows).set_index("name")


def _validate_return_series(returns: pd.Series) -> pd.Series:
    if returns.empty:
        raise RiskMetricError("Return series is empty")

    numeric_returns = pd.to_numeric(returns, errors="coerce")

    if numeric_returns.isna().any():
        raise RiskMetricError("Return series contains missing or non-numeric values")

    return pd.Series(numeric_returns, index=returns.index, name=returns.name)


def _validate_confidence_level(confidence_level: float) -> None:
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise RiskMetricError("Confidence level must be between 0 and 1")


def _validate_annualization_factor(annualization_factor: int) -> None:
    if annualization_factor <= 0:
        raise RiskMetricError("Annualization factor must be positive")
