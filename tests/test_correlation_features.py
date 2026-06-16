import pandas as pd
import pytest

from regime_risk_engine.features.correlation import (
    DEFAULT_CORRELATION_WINDOWS,
    CorrelationFeatureError,
    build_correlation_features,
    calculate_equity_bond_correlation,
    calculate_rolling_average_correlation,
    calculate_rolling_correlation_matrices,
    calculate_rolling_pairwise_correlations,
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
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
                "SPY",
                "SPY",
                "TLT",
                "TLT",
                "TLT",
                "TLT",
                "GLD",
                "GLD",
                "GLD",
                "GLD",
            ],
            "return": [
                0.01,
                0.02,
                0.03,
                0.04,
                -0.01,
                -0.02,
                -0.03,
                -0.04,
                0.02,
                0.01,
                0.00,
                -0.01,
            ],
        }
    )


def test_default_correlation_windows_exist() -> None:
    assert DEFAULT_CORRELATION_WINDOWS == {
        "short": 21,
        "medium": 63,
        "long": 126,
        "annual": 252,
    }


def test_calculate_rolling_correlation_matrices() -> None:
    returns = make_returns()

    matrices = calculate_rolling_correlation_matrices(returns, window=2)

    assert len(matrices) == 3

    first_matrix = matrices[pd.Timestamp("2020-01-02")]

    assert set(first_matrix.columns) == {"SPY", "TLT", "GLD"}
    assert set(first_matrix.index) == {"SPY", "TLT", "GLD"}
    assert first_matrix.loc["SPY", "SPY"] == pytest.approx(1.0)


def test_calculate_rolling_average_correlation() -> None:
    returns = make_returns()

    features = calculate_rolling_average_correlation(
        returns,
        windows={"two_day": 2},
    )

    assert set(features.columns) == {
        "date",
        "avg_pairwise_corr_two_day_2d",
    }

    assert pd.isna(features.iloc[0]["avg_pairwise_corr_two_day_2d"])
    assert features.iloc[1]["avg_pairwise_corr_two_day_2d"] == pytest.approx(-1.0 / 3.0)


def test_calculate_rolling_pairwise_correlations() -> None:
    returns = make_returns()

    features = calculate_rolling_pairwise_correlations(
        returns,
        pairs=[("SPY", "TLT")],
        windows={"three_day": 3},
    )

    assert set(features.columns) == {
        "date",
        "rolling_corr_SPY_TLT_three_day_3d",
    }

    assert pd.isna(features.iloc[0]["rolling_corr_SPY_TLT_three_day_3d"])
    assert pd.isna(features.iloc[1]["rolling_corr_SPY_TLT_three_day_3d"])
    assert features.iloc[2]["rolling_corr_SPY_TLT_three_day_3d"] == pytest.approx(-1.0)


def test_calculate_equity_bond_correlation() -> None:
    returns = make_returns()

    features = calculate_equity_bond_correlation(
        returns,
        equity_ticker="SPY",
        bond_ticker="TLT",
        windows={"three_day": 3},
    )

    assert "rolling_corr_SPY_TLT_three_day_3d" in features.columns


def test_build_correlation_features() -> None:
    returns = make_returns()

    features = build_correlation_features(
        returns,
        pairs=[("SPY", "TLT")],
        windows={"two_day": 2},
    )

    assert set(features.columns) == {
        "date",
        "avg_pairwise_corr_two_day_2d",
        "rolling_corr_SPY_TLT_two_day_2d",
    }
    assert len(features) == 4


def test_correlation_features_reject_missing_columns() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
        }
    )

    with pytest.raises(CorrelationFeatureError, match="Missing required return column"):
        calculate_rolling_average_correlation(returns)


def test_correlation_features_reject_empty_data() -> None:
    returns = pd.DataFrame(columns=["date", "ticker", "return"])

    with pytest.raises(CorrelationFeatureError, match="Return data is empty"):
        calculate_rolling_average_correlation(returns)


def test_correlation_features_reject_missing_return_values() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
            "return": [None],
        }
    )

    with pytest.raises(CorrelationFeatureError, match="missing values"):
        calculate_rolling_average_correlation(returns)


def test_correlation_features_reject_duplicate_date_ticker_rows() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "ticker": ["SPY", "SPY"],
            "return": [0.01, 0.02],
        }
    )

    with pytest.raises(CorrelationFeatureError, match="duplicate"):
        calculate_rolling_average_correlation(returns)


def test_correlation_features_reject_invalid_windows() -> None:
    returns = make_returns()

    with pytest.raises(CorrelationFeatureError, match="greater than 1"):
        calculate_rolling_average_correlation(returns, windows={"bad": 1})


def test_correlation_features_reject_unknown_pair_ticker() -> None:
    returns = make_returns()

    with pytest.raises(CorrelationFeatureError, match="unknown ticker"):
        calculate_rolling_pairwise_correlations(
            returns,
            pairs=[("SPY", "UNKNOWN")],
            windows={"two_day": 2},
        )


def test_correlation_features_reject_same_ticker_pair() -> None:
    returns = make_returns()

    with pytest.raises(CorrelationFeatureError, match="different tickers"):
        calculate_rolling_pairwise_correlations(
            returns,
            pairs=[("SPY", "SPY")],
            windows={"two_day": 2},
        )
