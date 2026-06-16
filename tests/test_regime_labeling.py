import pandas as pd
import pytest

from regime_risk_engine.regimes.labeling import (
    RegimeLabelingError,
    apply_regime_label_map,
    assign_regime_labels,
    build_labeled_regime_summary,
    calculate_regime_summary,
)


def make_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")

    return pd.DataFrame(
        {
            "date": list(dates) * 3,
            "ticker": ["SPY"] * 6 + ["TLT"] * 6 + ["GLD"] * 6,
            "return": [
                0.01,
                0.01,
                0.01,
                -0.05,
                0.04,
                -0.06,
                0.00,
                0.01,
                0.00,
                0.03,
                -0.02,
                0.04,
                0.01,
                0.00,
                0.01,
                -0.01,
                0.02,
                -0.02,
            ],
        }
    )


def make_regime_labels() -> pd.Series:
    return pd.Series(
        [0, 0, 0, 1, 1, 1],
        index=pd.date_range("2020-01-01", periods=6, freq="D"),
        name="regime",
    )


def test_calculate_regime_summary() -> None:
    returns = make_returns()
    labels = make_regime_labels()

    summary = calculate_regime_summary(
        returns,
        labels,
        annualization_factor=252,
    )

    assert set(summary.columns) == {
        "regime",
        "observation_count",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "average_pairwise_correlation",
    }
    assert summary["regime"].tolist() == [0, 1]
    assert summary["observation_count"].tolist() == [3, 3]


def test_assign_regime_labels() -> None:
    summary = pd.DataFrame(
        {
            "regime": [0, 1],
            "observation_count": [3, 3],
            "annualized_return": [0.20, -0.50],
            "annualized_volatility": [0.10, 0.80],
            "max_drawdown": [0.0, -0.20],
            "average_pairwise_correlation": [0.1, 0.8],
        }
    )

    label_map = assign_regime_labels(summary)

    assert label_map[0] == "bull_low_volatility"
    assert label_map[1] == "high_volatility_stress"


def test_apply_regime_label_map() -> None:
    labels = make_regime_labels()
    label_map = {
        0: "bull_low_volatility",
        1: "high_volatility_stress",
    }

    readable_labels = apply_regime_label_map(labels, label_map)

    assert readable_labels.name == "regime_label"
    assert readable_labels.index.equals(labels.index)
    assert readable_labels.iloc[0] == "bull_low_volatility"
    assert readable_labels.iloc[-1] == "high_volatility_stress"


def test_build_labeled_regime_summary() -> None:
    returns = make_returns()
    labels = make_regime_labels()

    summary = build_labeled_regime_summary(
        returns,
        labels,
        annualization_factor=252,
    )

    assert "regime_label" in summary.columns
    assert len(summary) == 2
    assert summary["regime_label"].notna().all()


def test_regime_summary_rejects_missing_return_columns() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
        }
    )
    labels = make_regime_labels()

    with pytest.raises(RegimeLabelingError, match="Missing required return column"):
        calculate_regime_summary(returns, labels)


def test_regime_summary_rejects_empty_returns() -> None:
    returns = pd.DataFrame(columns=["date", "ticker", "return"])
    labels = make_regime_labels()

    with pytest.raises(RegimeLabelingError, match="Return data is empty"):
        calculate_regime_summary(returns, labels)


def test_regime_summary_rejects_non_datetime_label_index() -> None:
    returns = make_returns()
    labels = pd.Series([0, 1], index=[0, 1], name="regime")

    with pytest.raises(RegimeLabelingError, match="DatetimeIndex"):
        calculate_regime_summary(returns, labels)


def test_regime_summary_rejects_no_overlapping_dates() -> None:
    returns = make_returns()
    labels = pd.Series(
        [0, 1],
        index=pd.to_datetime(["2030-01-01", "2030-01-02"]),
        name="regime",
    )

    with pytest.raises(RegimeLabelingError, match="no overlapping dates"):
        calculate_regime_summary(returns, labels)


def test_apply_regime_label_map_rejects_missing_label() -> None:
    labels = make_regime_labels()
    label_map = {
        0: "bull_low_volatility",
    }

    with pytest.raises(RegimeLabelingError, match="Missing human-readable label"):
        apply_regime_label_map(labels, label_map)


def test_assign_regime_labels_rejects_missing_summary_columns() -> None:
    summary = pd.DataFrame(
        {
            "regime": [0, 1],
            "annualized_return": [0.1, -0.1],
        }
    )

    with pytest.raises(RegimeLabelingError, match="Missing regime summary column"):
        assign_regime_labels(summary)
