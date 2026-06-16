import math

import pandas as pd
import pytest

from regime_risk_engine.risk.metrics import (
    RiskMetricError,
    annualized_return,
    annualized_volatility,
    cumulative_return,
    historical_cvar,
    historical_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize_multiple_return_series,
    summarize_risk_metrics,
)


def make_returns() -> pd.Series:
    return pd.Series(
        [0.10, -0.05, 0.02, -0.10, 0.04],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
        name="portfolio_return",
    )


def test_cumulative_return() -> None:
    returns = pd.Series([0.10, -0.05, 0.02])

    result = cumulative_return(returns)

    assert result == pytest.approx((1.10 * 0.95 * 1.02) - 1.0)


def test_annualized_return() -> None:
    returns = pd.Series([0.01, 0.01])

    result = annualized_return(returns, annualization_factor=2)

    assert result == pytest.approx(0.0201)


def test_annualized_volatility() -> None:
    returns = pd.Series([0.01, 0.03])

    result = annualized_volatility(returns, annualization_factor=2)

    expected = returns.std(ddof=1) * math.sqrt(2)

    assert result == pytest.approx(expected)


def test_sharpe_ratio() -> None:
    returns = pd.Series([0.01, 0.03, 0.02])

    result = sharpe_ratio(
        returns,
        risk_free_rate=0.0,
        annualization_factor=3,
    )

    expected = annualized_return(
        returns,
        annualization_factor=3,
    ) / annualized_volatility(
        returns,
        annualization_factor=3,
    )

    assert result == pytest.approx(expected)


def test_sortino_ratio() -> None:
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])

    result = sortino_ratio(
        returns,
        target_return=0.0,
        annualization_factor=4,
    )

    downside_returns = pd.Series([-0.01, -0.02])
    downside_deviation = downside_returns.std(ddof=1) * math.sqrt(4)
    expected = annualized_return(returns, annualization_factor=4) / downside_deviation

    assert result == pytest.approx(expected)


def test_max_drawdown() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])

    result = max_drawdown(returns)

    assert result == pytest.approx(-0.20)


def test_historical_var() -> None:
    returns = pd.Series([-0.10, -0.05, 0.00, 0.02, 0.04])

    result = historical_var(returns, confidence_level=0.80)

    assert result == pytest.approx(0.06)


def test_historical_cvar() -> None:
    returns = pd.Series([-0.10, -0.05, 0.00, 0.02, 0.04])

    result = historical_cvar(returns, confidence_level=0.80)

    assert result == pytest.approx(0.10)


def test_summarize_risk_metrics() -> None:
    returns = make_returns()

    summary = summarize_risk_metrics(
        returns,
        confidence_level=0.80,
        annualization_factor=252,
    )

    assert set(summary) == {
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "var",
        "cvar",
    }


def test_summarize_multiple_return_series() -> None:
    returns = make_returns()

    summary = summarize_multiple_return_series(
        {
            "static": returns,
            "dynamic": returns * 0.8,
        },
        confidence_level=0.80,
    )

    assert list(summary.index) == ["static", "dynamic"]
    assert "sharpe_ratio" in summary.columns


def test_metrics_reject_empty_returns() -> None:
    returns = pd.Series(dtype=float)

    with pytest.raises(RiskMetricError, match="empty"):
        cumulative_return(returns)


def test_metrics_reject_missing_returns() -> None:
    returns = pd.Series([0.01, None])

    with pytest.raises(RiskMetricError, match="missing"):
        cumulative_return(returns)


def test_metrics_reject_invalid_confidence_level() -> None:
    returns = make_returns()

    with pytest.raises(RiskMetricError, match="Confidence level"):
        historical_var(returns, confidence_level=1.0)


def test_metrics_reject_invalid_annualization_factor() -> None:
    returns = make_returns()

    with pytest.raises(RiskMetricError, match="Annualization factor"):
        annualized_return(returns, annualization_factor=0)


def test_summarize_multiple_return_series_rejects_empty_mapping() -> None:
    with pytest.raises(RiskMetricError, match="At least one"):
        summarize_multiple_return_series({})
