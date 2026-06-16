import pandas as pd
import pytest

from regime_risk_engine.risk.regime_summary import (
    RegimeRiskSummary,
    RegimeRiskSummaryError,
    build_regime_risk_summary,
    calculate_portfolio_returns,
    calculate_regime_asset_risk_summary,
    calculate_regime_portfolio_risk_summary,
)


def make_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")

    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["SPY"] * 6 + ["TLT"] * 6,
            "return": [
                0.01,
                0.02,
                -0.01,
                -0.04,
                0.03,
                -0.02,
                0.00,
                0.01,
                0.02,
                0.03,
                -0.01,
                0.01,
            ],
        }
    )


def make_regime_labels() -> pd.Series:
    return pd.Series(
        [0, 0, 0, 1, 1, 1],
        index=pd.date_range("2020-01-01", periods=6, freq="D"),
        name="regime",
    )


def test_calculate_equal_weight_portfolio_returns() -> None:
    returns = make_returns()

    portfolio_returns = calculate_portfolio_returns(returns)

    assert portfolio_returns.name == "portfolio_return"
    assert len(portfolio_returns) == 6
    assert portfolio_returns.iloc[0] == pytest.approx(0.005)


def test_calculate_weighted_portfolio_returns() -> None:
    returns = make_returns()

    portfolio_returns = calculate_portfolio_returns(
        returns,
        weights={
            "SPY": 0.6,
            "TLT": 0.4,
        },
    )

    expected_first_return = (0.01 * 0.6) + (0.00 * 0.4)

    assert portfolio_returns.iloc[0] == pytest.approx(expected_first_return)


def test_calculate_regime_portfolio_risk_summary() -> None:
    returns = make_returns()
    labels = make_regime_labels()

    summary = calculate_regime_portfolio_risk_summary(
        returns,
        labels,
        confidence_level=0.80,
        annualization_factor=252,
    )

    assert summary["regime"].tolist() == [0, 1]
    assert summary["observation_count"].tolist() == [3, 3]
    assert "annualized_return" in summary.columns
    assert "annualized_volatility" in summary.columns
    assert "max_drawdown" in summary.columns
    assert "var" in summary.columns
    assert "cvar" in summary.columns


def test_calculate_regime_asset_risk_summary() -> None:
    returns = make_returns()
    labels = make_regime_labels()

    summary = calculate_regime_asset_risk_summary(
        returns,
        labels,
        confidence_level=0.80,
        annualization_factor=252,
    )

    assert set(summary["regime"]) == {0, 1}
    assert set(summary["ticker"]) == {"SPY", "TLT"}
    assert len(summary) == 4
    assert "sharpe_ratio" in summary.columns


def test_build_regime_risk_summary() -> None:
    returns = make_returns()
    labels = make_regime_labels()

    summary = build_regime_risk_summary(
        returns,
        labels,
        weights={
            "SPY": 0.5,
            "TLT": 0.5,
        },
        confidence_level=0.80,
    )

    assert isinstance(summary, RegimeRiskSummary)
    assert len(summary.portfolio_summary) == 2
    assert len(summary.asset_summary) == 4


def test_regime_risk_summary_rejects_missing_return_columns() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
        }
    )
    labels = make_regime_labels()

    with pytest.raises(RegimeRiskSummaryError, match="Missing required return"):
        calculate_regime_portfolio_risk_summary(returns, labels)


def test_regime_risk_summary_rejects_empty_returns() -> None:
    returns = pd.DataFrame(columns=["date", "ticker", "return"])
    labels = make_regime_labels()

    with pytest.raises(RegimeRiskSummaryError, match="Return data is empty"):
        calculate_regime_portfolio_risk_summary(returns, labels)


def test_regime_risk_summary_rejects_non_datetime_labels() -> None:
    returns = make_returns()
    labels = pd.Series([0, 1], index=[0, 1], name="regime")

    with pytest.raises(RegimeRiskSummaryError, match="DatetimeIndex"):
        calculate_regime_portfolio_risk_summary(returns, labels)


def test_regime_risk_summary_rejects_no_overlapping_dates() -> None:
    returns = make_returns()
    labels = pd.Series(
        [0, 1],
        index=pd.to_datetime(["2030-01-01", "2030-01-02"]),
        name="regime",
    )

    with pytest.raises(RegimeRiskSummaryError, match="no overlapping dates"):
        calculate_regime_portfolio_risk_summary(returns, labels)


def test_calculate_portfolio_returns_rejects_missing_weights() -> None:
    returns = make_returns()

    with pytest.raises(RegimeRiskSummaryError, match="Missing weight"):
        calculate_portfolio_returns(
            returns,
            weights={
                "SPY": 1.0,
            },
        )


def test_calculate_portfolio_returns_rejects_weights_that_do_not_sum_to_one() -> None:
    returns = make_returns()

    with pytest.raises(RegimeRiskSummaryError, match="sum to 1.0"):
        calculate_portfolio_returns(
            returns,
            weights={
                "SPY": 0.7,
                "TLT": 0.7,
            },
        )


def test_calculate_portfolio_returns_rejects_negative_weights() -> None:
    returns = make_returns()

    with pytest.raises(RegimeRiskSummaryError, match="non-negative"):
        calculate_portfolio_returns(
            returns,
            weights={
                "SPY": 1.1,
                "TLT": -0.1,
            },
        )
