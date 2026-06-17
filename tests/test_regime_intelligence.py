import pandas as pd
import pytest

from regime_risk_engine.research.regime_intelligence import (
    RegimeIntelligenceError,
    RegimeIntelligenceSummary,
    build_regime_intelligence_summary,
)


def make_asset_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    rows = []

    for index, _date in enumerate(dates):
        if index < 30:
            rows.append(
                {
                    "SPY": 0.004,
                    "TLT": 0.0005,
                    "GLD": 0.0002,
                }
            )
        elif index < 60:
            rows.append(
                {
                    "SPY": -0.009,
                    "TLT": 0.002,
                    "GLD": 0.001,
                }
            )
        else:
            rows.append(
                {
                    "SPY": 0.0005,
                    "TLT": -0.002,
                    "GLD": 0.004,
                }
            )

    return pd.DataFrame(rows, index=dates)


def make_regime_labels() -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=90, freq="D")

    return pd.Series(
        [0] * 30 + [1] * 30 + [2] * 30,
        index=dates,
        name="regime",
    )


def test_build_regime_intelligence_summary() -> None:
    summary = build_regime_intelligence_summary(
        asset_returns=make_asset_returns(),
        regime_labels=make_regime_labels(),
    )

    assert isinstance(summary, RegimeIntelligenceSummary)
    assert len(summary.profile_table) == 3
    assert summary.strongest_regime == 0
    assert summary.weakest_regime == 1
    assert summary.stress_regime == 1
    assert "interpretable market states" in summary.narrative

    labels = set(summary.profile_table["label"])

    assert "Growth / risk-on" in labels
    assert "Defensive / stress" in labels
    assert "Inflation / real assets" in labels


def test_regime_intelligence_profile_contains_market_diagnostics() -> None:
    summary = build_regime_intelligence_summary(
        asset_returns=make_asset_returns(),
        regime_labels=make_regime_labels(),
    )

    expected_columns = {
        "regime",
        "observation_count",
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_correlation",
        "equity_return",
        "defensive_return",
        "real_asset_return",
        "best_asset",
        "worst_asset",
        "label",
        "interpretation",
    }

    assert expected_columns.issubset(summary.profile_table.columns)


def test_regime_intelligence_accepts_custom_asset_roles() -> None:
    asset_returns = make_asset_returns().rename(
        columns={
            "SPY": "EQUITY_FUND",
            "TLT": "BOND_FUND",
            "GLD": "GOLD_FUND",
        }
    )

    summary = build_regime_intelligence_summary(
        asset_returns=asset_returns,
        regime_labels=make_regime_labels(),
        asset_roles={
            "EQUITY_FUND": "equity",
            "BOND_FUND": "defensive",
            "GOLD_FUND": "real_asset",
        },
    )

    labels = set(summary.profile_table["label"])

    assert "Growth / risk-on" in labels
    assert "Defensive / stress" in labels
    assert "Inflation / real assets" in labels


def test_regime_intelligence_rejects_empty_returns() -> None:
    with pytest.raises(RegimeIntelligenceError, match="Asset returns"):
        build_regime_intelligence_summary(
            asset_returns=pd.DataFrame(),
            regime_labels=make_regime_labels(),
        )


def test_regime_intelligence_rejects_empty_regimes() -> None:
    with pytest.raises(RegimeIntelligenceError, match="Regime labels"):
        build_regime_intelligence_summary(
            asset_returns=make_asset_returns(),
            regime_labels=pd.Series(dtype=int),
        )


def test_regime_intelligence_rejects_non_numeric_returns() -> None:
    asset_returns = make_asset_returns()
    asset_returns["SPY"] = asset_returns["SPY"].astype(object)
    asset_returns.loc[asset_returns.index[0], "SPY"] = "bad"

    with pytest.raises(RegimeIntelligenceError, match="numeric"):
        build_regime_intelligence_summary(
            asset_returns=asset_returns,
            regime_labels=make_regime_labels(),
        )


def test_regime_intelligence_rejects_unknown_asset_role_override() -> None:
    with pytest.raises(RegimeIntelligenceError, match="unknown ticker"):
        build_regime_intelligence_summary(
            asset_returns=make_asset_returns(),
            regime_labels=make_regime_labels(),
            asset_roles={
                "UNKNOWN": "equity",
            },
        )


def test_regime_intelligence_rejects_single_regime() -> None:
    labels = pd.Series(
        [0] * 90,
        index=make_regime_labels().index,
        name="regime",
    )

    with pytest.raises(RegimeIntelligenceError, match="At least two regimes"):
        build_regime_intelligence_summary(
            asset_returns=make_asset_returns(),
            regime_labels=labels,
        )
