import pandas as pd
import pytest

from regime_risk_engine.features.trend import (
    DEFAULT_TREND_WINDOWS,
    TrendFeatureError,
    build_trend_features,
    calculate_drawdown_features,
    calculate_momentum_features,
    calculate_moving_average_distance_features,
)


def make_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
                "SPY",
                "SPY",
                "SPY",
                "TLT",
                "TLT",
                "TLT",
                "TLT",
                "TLT",
            ],
            "return": [
                0.01,
                0.02,
                -0.01,
                0.03,
                -0.02,
                0.00,
                0.01,
                0.01,
                -0.02,
                0.00,
            ],
        }
    )


def test_default_trend_windows_exist() -> None:
    assert DEFAULT_TREND_WINDOWS == {
        "short": 21,
        "medium": 63,
        "long": 126,
        "annual": 252,
    }


def test_calculate_momentum_features() -> None:
    returns = make_returns()

    features = calculate_momentum_features(returns, windows={"two_day": 2})

    assert set(features.columns) == {
        "date",
        "ticker",
        "momentum_two_day_2d",
    }

    spy_features = features[features["ticker"] == "SPY"]

    assert pd.isna(spy_features.iloc[0]["momentum_two_day_2d"])
    assert spy_features.iloc[1]["momentum_two_day_2d"] == pytest.approx(
        (1.01 * 1.02) - 1.0
    )


def test_calculate_moving_average_distance_features() -> None:
    returns = make_returns()

    features = calculate_moving_average_distance_features(
        returns,
        windows={"two_day": 2},
    )

    assert set(features.columns) == {
        "date",
        "ticker",
        "ma_distance_two_day_2d",
    }

    spy_features = features[features["ticker"] == "SPY"]

    first_index = 1.01
    second_index = 1.01 * 1.02
    expected_ma = (first_index + second_index) / 2
    expected_distance = second_index / expected_ma - 1.0

    assert pd.isna(spy_features.iloc[0]["ma_distance_two_day_2d"])
    assert spy_features.iloc[1]["ma_distance_two_day_2d"] == pytest.approx(
        expected_distance
    )


def test_calculate_drawdown_features() -> None:
    returns = make_returns()

    features = calculate_drawdown_features(returns)

    assert set(features.columns) == {
        "date",
        "ticker",
        "drawdown",
    }

    spy_features = features[features["ticker"] == "SPY"]

    assert spy_features.iloc[0]["drawdown"] == pytest.approx(0.0)

    third_index = 1.01 * 1.02 * 0.99
    running_peak = 1.01 * 1.02
    expected_drawdown = third_index / running_peak - 1.0

    assert spy_features.iloc[2]["drawdown"] == pytest.approx(expected_drawdown)


def test_build_trend_features_combines_all_trend_features() -> None:
    returns = make_returns()

    features = build_trend_features(returns, windows={"two_day": 2})

    assert set(features.columns) == {
        "date",
        "ticker",
        "momentum_two_day_2d",
        "ma_distance_two_day_2d",
        "drawdown",
    }
    assert len(features) == len(returns)


def test_trend_features_sort_by_ticker_and_date() -> None:
    returns = make_returns().sample(frac=1.0, random_state=42)

    features = calculate_momentum_features(returns, windows={"two_day": 2})
    spy_features = features[features["ticker"] == "SPY"]

    assert spy_features["date"].tolist() == sorted(spy_features["date"].tolist())


def test_trend_features_reject_missing_columns() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
        }
    )

    with pytest.raises(TrendFeatureError, match="Missing required return column"):
        calculate_momentum_features(returns)


def test_trend_features_reject_empty_data() -> None:
    returns = pd.DataFrame(columns=["date", "ticker", "return"])

    with pytest.raises(TrendFeatureError, match="Return data is empty"):
        calculate_momentum_features(returns)


def test_trend_features_reject_missing_return_values() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
            "return": [None],
        }
    )

    with pytest.raises(TrendFeatureError, match="missing values"):
        calculate_momentum_features(returns)


def test_trend_features_reject_invalid_windows() -> None:
    returns = make_returns()

    with pytest.raises(TrendFeatureError, match="greater than 1"):
        calculate_momentum_features(returns, windows={"bad": 1})
