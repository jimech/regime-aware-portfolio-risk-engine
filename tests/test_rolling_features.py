import pandas as pd
import pytest

from regime_risk_engine.features.rolling import (
    DEFAULT_ROLLING_WINDOWS,
    FeatureEngineeringError,
    build_rolling_features,
    calculate_rolling_returns,
    calculate_rolling_volatility,
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
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                ]
            ),
            "ticker": ["SPY", "SPY", "SPY", "SPY", "TLT", "TLT", "TLT", "TLT"],
            "return": [0.01, 0.02, -0.01, 0.03, 0.00, 0.01, 0.01, -0.02],
        }
    )


def test_default_rolling_windows_exist() -> None:
    assert DEFAULT_ROLLING_WINDOWS == {
        "short": 21,
        "medium": 63,
        "long": 126,
        "annual": 252,
    }


def test_calculate_rolling_returns() -> None:
    returns = make_returns()

    features = calculate_rolling_returns(returns, windows={"two_day": 2})

    assert set(features.columns) == {
        "date",
        "ticker",
        "rolling_return_two_day_2d",
    }

    spy_features = features[features["ticker"] == "SPY"]

    assert pd.isna(spy_features.iloc[0]["rolling_return_two_day_2d"])
    assert spy_features.iloc[1]["rolling_return_two_day_2d"] == pytest.approx(
        (1.01 * 1.02) - 1.0
    )


def test_calculate_rolling_volatility() -> None:
    returns = make_returns()

    features = calculate_rolling_volatility(
        returns,
        windows={"two_day": 2},
        annualization_factor=252,
    )

    assert set(features.columns) == {
        "date",
        "ticker",
        "rolling_volatility_two_day_2d",
    }

    spy_features = features[features["ticker"] == "SPY"]

    assert pd.isna(spy_features.iloc[0]["rolling_volatility_two_day_2d"])
    assert spy_features.iloc[1]["rolling_volatility_two_day_2d"] > 0


def test_build_rolling_features_combines_return_and_volatility_features() -> None:
    returns = make_returns()

    features = build_rolling_features(returns, windows={"two_day": 2})

    assert set(features.columns) == {
        "date",
        "ticker",
        "rolling_return_two_day_2d",
        "rolling_volatility_two_day_2d",
    }
    assert len(features) == len(returns)


def test_rolling_features_sort_by_ticker_and_date() -> None:
    returns = make_returns().sample(frac=1.0, random_state=42)

    features = calculate_rolling_returns(returns, windows={"two_day": 2})

    spy_features = features[features["ticker"] == "SPY"]

    assert spy_features["date"].tolist() == sorted(spy_features["date"].tolist())


def test_rolling_features_reject_missing_columns() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
        }
    )

    with pytest.raises(FeatureEngineeringError, match="Missing required return column"):
        calculate_rolling_returns(returns)


def test_rolling_features_reject_empty_data() -> None:
    returns = pd.DataFrame(columns=["date", "ticker", "return"])

    with pytest.raises(FeatureEngineeringError, match="Return data is empty"):
        calculate_rolling_returns(returns)


def test_rolling_features_reject_missing_return_values() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
            "return": [None],
        }
    )

    with pytest.raises(FeatureEngineeringError, match="missing values"):
        calculate_rolling_returns(returns)


def test_rolling_features_reject_invalid_windows() -> None:
    returns = make_returns()

    with pytest.raises(FeatureEngineeringError, match="greater than 1"):
        calculate_rolling_returns(returns, windows={"bad": 1})


def test_rolling_volatility_rejects_invalid_annualization_factor() -> None:
    returns = make_returns()

    with pytest.raises(FeatureEngineeringError, match="Annualization factor"):
        calculate_rolling_volatility(returns, annualization_factor=0)
