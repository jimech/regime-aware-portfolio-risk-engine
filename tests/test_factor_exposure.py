import pandas as pd
import pytest

from regime_risk_engine.research.factor_exposure import (
    FactorExposureError,
    FactorExposureSummary,
    build_factor_exposure_summary,
)


def make_factor_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=80, freq="D")
    rows = []

    for index, _date in enumerate(dates):
        equity_factor = 0.002 if index < 40 else -0.001
        defensive_factor = -0.0005 if index < 40 else 0.0015
        real_asset_factor = 0.0003 if index < 55 else 0.002

        rows.append(
            {
                "equity": equity_factor,
                "defensive": defensive_factor,
                "real_asset": real_asset_factor,
            }
        )

    return pd.DataFrame(rows, index=dates)


def make_strategy_returns() -> pd.DataFrame:
    factors = make_factor_returns()

    static = (
        0.70 * factors["equity"]
        + 0.20 * factors["defensive"]
        + 0.10 * factors["real_asset"]
        + 0.0001
    )
    dynamic = (
        0.40 * factors["equity"]
        + 0.35 * factors["defensive"]
        + 0.25 * factors["real_asset"]
        + 0.0002
    )

    return pd.DataFrame(
        {
            "static": static,
            "dynamic": dynamic,
        },
        index=factors.index,
    )


def make_regime_labels() -> pd.Series:
    dates = make_factor_returns().index

    return pd.Series(
        [0] * 40 + [1] * 40,
        index=dates,
        name="regime",
    )


def test_build_factor_exposure_summary() -> None:
    summary = build_factor_exposure_summary(
        strategy_returns=make_strategy_returns(),
        factor_returns=make_factor_returns(),
        regime_labels=make_regime_labels(),
    )

    assert isinstance(summary, FactorExposureSummary)
    assert len(summary.exposure_table) == 2
    assert summary.regime_exposure_table is not None
    assert len(summary.regime_exposure_table) == 4
    assert len(summary.dominant_factor_by_strategy) == 2
    assert "Factor exposure analysis estimated" in summary.narrative


def test_factor_exposure_estimates_expected_betas() -> None:
    summary = build_factor_exposure_summary(
        strategy_returns=make_strategy_returns(),
        factor_returns=make_factor_returns(),
    )

    exposure = summary.exposure_table.set_index("strategy")

    assert (
        exposure.loc["static", "equity_beta"]
        > exposure.loc[
            "dynamic",
            "equity_beta",
        ]
    )
    assert (
        exposure.loc["dynamic", "defensive_beta"]
        > exposure.loc[
            "static",
            "defensive_beta",
        ]
    )
    assert (
        exposure.loc["dynamic", "real_asset_beta"]
        > exposure.loc[
            "static",
            "real_asset_beta",
        ]
    )


def test_factor_exposure_identifies_dominant_factor() -> None:
    summary = build_factor_exposure_summary(
        strategy_returns=make_strategy_returns(),
        factor_returns=make_factor_returns(),
    )

    dominant = summary.dominant_factor_by_strategy.set_index("strategy")

    assert dominant.loc["static", "dominant_factor"] == "equity"
    assert dominant.loc["dynamic", "dominant_factor"] in {
        "equity",
        "defensive",
        "real_asset",
    }


def test_factor_exposure_works_without_regimes() -> None:
    summary = build_factor_exposure_summary(
        strategy_returns=make_strategy_returns(),
        factor_returns=make_factor_returns(),
    )

    assert summary.regime_exposure_table is None


def test_factor_exposure_rejects_empty_strategy_returns() -> None:
    with pytest.raises(FactorExposureError, match="Strategy returns"):
        build_factor_exposure_summary(
            strategy_returns=pd.DataFrame(),
            factor_returns=make_factor_returns(),
        )


def test_factor_exposure_rejects_empty_factor_returns() -> None:
    with pytest.raises(FactorExposureError, match="Factor returns"):
        build_factor_exposure_summary(
            strategy_returns=make_strategy_returns(),
            factor_returns=pd.DataFrame(),
        )


def test_factor_exposure_rejects_non_numeric_factor_returns() -> None:
    factor_returns = make_factor_returns()
    factor_returns["equity"] = factor_returns["equity"].astype(object)
    factor_returns.loc[factor_returns.index[0], "equity"] = "bad"

    with pytest.raises(FactorExposureError, match="numeric"):
        build_factor_exposure_summary(
            strategy_returns=make_strategy_returns(),
            factor_returns=factor_returns,
        )


def test_factor_exposure_rejects_no_overlap() -> None:
    factor_returns = make_factor_returns()
    factor_returns.index = factor_returns.index + pd.Timedelta(days=365)

    with pytest.raises(FactorExposureError, match="overlapping"):
        build_factor_exposure_summary(
            strategy_returns=make_strategy_returns(),
            factor_returns=factor_returns,
        )


def test_factor_exposure_rejects_too_few_observations() -> None:
    strategy_returns = make_strategy_returns().iloc[:3]
    factor_returns = make_factor_returns().iloc[:3]

    with pytest.raises(FactorExposureError, match="Not enough observations"):
        build_factor_exposure_summary(
            strategy_returns=strategy_returns,
            factor_returns=factor_returns,
        )
