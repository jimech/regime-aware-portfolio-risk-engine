import pandas as pd
import pytest

from regime_risk_engine.research.attribution import (
    StrategyAttributionError,
    StrategyAttributionSummary,
    build_strategy_attribution_summary,
)


def make_asset_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")

    rows = []

    for index, _date in enumerate(dates):
        if index < 3:
            rows.append({"SPY": 0.01, "TLT": 0.0, "GLD": 0.0})
        elif index < 6:
            rows.append({"SPY": -0.01, "TLT": 0.01, "GLD": 0.0})
        else:
            rows.append({"SPY": 0.0, "TLT": -0.005, "GLD": 0.01})

    return pd.DataFrame(rows, index=dates)


def make_static_weight_frame() -> pd.DataFrame:
    dates = make_asset_returns().index

    return pd.DataFrame(
        {
            "SPY": [0.60] * len(dates),
            "TLT": [0.30] * len(dates),
            "GLD": [0.10] * len(dates),
        },
        index=dates,
    )


def make_dynamic_weight_frame() -> pd.DataFrame:
    dates = make_asset_returns().index
    rows = []

    for index, _date in enumerate(dates):
        if index < 3:
            rows.append({"SPY": 0.80, "TLT": 0.10, "GLD": 0.10})
        elif index < 6:
            rows.append({"SPY": 0.20, "TLT": 0.70, "GLD": 0.10})
        else:
            rows.append({"SPY": 0.05, "TLT": 0.20, "GLD": 0.75})

    return pd.DataFrame(rows, index=dates)


def make_regime_labels() -> pd.Series:
    dates = make_asset_returns().index

    return pd.Series(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
        index=dates,
        name="regime",
    )


def test_build_strategy_attribution_summary() -> None:
    summary = build_strategy_attribution_summary(
        asset_returns=make_asset_returns(),
        static_weight_frame=make_static_weight_frame(),
        dynamic_weight_frame=make_dynamic_weight_frame(),
        regime_labels=make_regime_labels(),
    )

    assert isinstance(summary, StrategyAttributionSummary)
    assert len(summary.asset_attribution) == 3
    assert summary.regime_attribution is not None
    assert len(summary.regime_attribution) == 3
    assert summary.top_positive_asset == "GLD"
    assert summary.top_negative_asset is None
    assert summary.strongest_regime == 2
    assert "Total active contribution" in summary.narrative


def test_asset_attribution_contains_expected_columns() -> None:
    summary = build_strategy_attribution_summary(
        asset_returns=make_asset_returns(),
        static_weight_frame=make_static_weight_frame(),
        dynamic_weight_frame=make_dynamic_weight_frame(),
        regime_labels=make_regime_labels(),
    )

    expected_columns = {
        "asset",
        "average_static_weight",
        "average_dynamic_weight",
        "average_active_weight",
        "static_return_contribution",
        "dynamic_return_contribution",
        "active_return_contribution",
    }

    assert expected_columns.issubset(summary.asset_attribution.columns)


def test_strategy_attribution_works_without_regimes() -> None:
    summary = build_strategy_attribution_summary(
        asset_returns=make_asset_returns(),
        static_weight_frame=make_static_weight_frame(),
        dynamic_weight_frame=make_dynamic_weight_frame(),
    )

    assert summary.regime_attribution is None
    assert summary.strongest_regime is None
    assert summary.weakest_regime is None


def test_strategy_attribution_identifies_negative_contributor() -> None:
    dynamic_weights = make_dynamic_weight_frame()
    dynamic_weights["SPY"] = 0.90
    dynamic_weights["TLT"] = 0.05
    dynamic_weights["GLD"] = 0.05

    summary = build_strategy_attribution_summary(
        asset_returns=make_asset_returns(),
        static_weight_frame=make_static_weight_frame(),
        dynamic_weight_frame=dynamic_weights,
        regime_labels=make_regime_labels(),
    )

    assert summary.top_negative_asset is not None


def test_strategy_attribution_rejects_empty_returns() -> None:
    with pytest.raises(StrategyAttributionError, match="Asset returns"):
        build_strategy_attribution_summary(
            asset_returns=pd.DataFrame(),
            static_weight_frame=make_static_weight_frame(),
            dynamic_weight_frame=make_dynamic_weight_frame(),
        )


def test_strategy_attribution_rejects_mismatched_columns() -> None:
    static_weights = make_static_weight_frame().drop(columns=["GLD"])

    with pytest.raises(StrategyAttributionError, match="columns"):
        build_strategy_attribution_summary(
            asset_returns=make_asset_returns(),
            static_weight_frame=static_weights,
            dynamic_weight_frame=make_dynamic_weight_frame(),
        )


def test_strategy_attribution_rejects_non_numeric_returns() -> None:
    asset_returns = make_asset_returns()
    asset_returns["SPY"] = asset_returns["SPY"].astype(object)
    asset_returns.loc[asset_returns.index[0], "SPY"] = "bad"

    with pytest.raises(StrategyAttributionError, match="numeric"):
        build_strategy_attribution_summary(
            asset_returns=asset_returns,
            static_weight_frame=make_static_weight_frame(),
            dynamic_weight_frame=make_dynamic_weight_frame(),
        )


def test_strategy_attribution_rejects_non_overlapping_dates() -> None:
    dynamic_weights = make_dynamic_weight_frame()
    dynamic_weights.index = dynamic_weights.index + pd.Timedelta(days=100)

    with pytest.raises(StrategyAttributionError, match="overlapping"):
        build_strategy_attribution_summary(
            asset_returns=make_asset_returns(),
            static_weight_frame=make_static_weight_frame(),
            dynamic_weight_frame=dynamic_weights,
        )
