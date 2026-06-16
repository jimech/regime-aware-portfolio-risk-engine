import pandas as pd
import pytest

from regime_risk_engine.risk.correlation import (
    RegimeCorrelationAnalytics,
    RegimeCorrelationError,
    build_regime_correlation_analytics,
    calculate_regime_average_correlation_summary,
    calculate_regime_correlation_matrices,
    calculate_regime_covariance_matrices,
)


def make_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=6, freq="D")

    return pd.DataFrame(
        {
            "date": list(dates) * 3,
            "ticker": ["SPY"] * 6 + ["TLT"] * 6 + ["GLD"] * 6,
            "return": [
                0.01,
                0.02,
                0.03,
                -0.01,
                -0.02,
                -0.03,
                -0.01,
                -0.02,
                -0.03,
                0.01,
                0.02,
                0.03,
                0.02,
                0.01,
                0.00,
                -0.02,
                -0.01,
                0.00,
            ],
        }
    )


def make_regime_labels() -> pd.Series:
    return pd.Series(
        [0, 0, 0, 1, 1, 1],
        index=pd.date_range("2020-01-01", periods=6, freq="D"),
        name="regime",
    )


def test_calculate_regime_correlation_matrices() -> None:
    returns = make_returns()
    labels = make_regime_labels()

    matrices = calculate_regime_correlation_matrices(returns, labels)

    assert set(matrices) == {0, 1}

    first_matrix = matrices[0]

    assert first_matrix.shape == (3, 3)
    assert set(first_matrix.columns) == {"SPY", "TLT", "GLD"}
    assert set(first_matrix.index) == {"SPY", "TLT", "GLD"}
    assert first_matrix.loc["SPY", "SPY"] == pytest.approx(1.0)


def test_calculate_regime_covariance_matrices() -> None:
    returns = make_returns()
    labels = make_regime_labels()

    matrices = calculate_regime_covariance_matrices(
        returns,
        labels,
        annualization_factor=252,
    )

    assert set(matrices) == {0, 1}

    first_matrix = matrices[0]

    assert first_matrix.shape == (3, 3)
    assert set(first_matrix.columns) == {"SPY", "TLT", "GLD"}
    assert set(first_matrix.index) == {"SPY", "TLT", "GLD"}


def test_calculate_regime_average_correlation_summary() -> None:
    returns = make_returns()
    labels = make_regime_labels()

    summary = calculate_regime_average_correlation_summary(returns, labels)

    assert summary["regime"].tolist() == [0, 1]
    assert summary["asset_count"].tolist() == [3, 3]
    assert "average_pairwise_correlation" in summary.columns


def test_build_regime_correlation_analytics() -> None:
    returns = make_returns()
    labels = make_regime_labels()

    analytics = build_regime_correlation_analytics(returns, labels)

    assert isinstance(analytics, RegimeCorrelationAnalytics)
    assert set(analytics.correlation_matrices) == {0, 1}
    assert set(analytics.covariance_matrices) == {0, 1}
    assert len(analytics.summary) == 2


def test_regime_correlation_rejects_missing_return_columns() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
        }
    )
    labels = make_regime_labels()

    with pytest.raises(RegimeCorrelationError, match="Missing required return"):
        calculate_regime_correlation_matrices(returns, labels)


def test_regime_correlation_rejects_empty_returns() -> None:
    returns = pd.DataFrame(columns=["date", "ticker", "return"])
    labels = make_regime_labels()

    with pytest.raises(RegimeCorrelationError, match="Return data is empty"):
        calculate_regime_correlation_matrices(returns, labels)


def test_regime_correlation_rejects_non_datetime_labels() -> None:
    returns = make_returns()
    labels = pd.Series([0, 1], index=[0, 1], name="regime")

    with pytest.raises(RegimeCorrelationError, match="DatetimeIndex"):
        calculate_regime_correlation_matrices(returns, labels)


def test_regime_correlation_rejects_no_overlapping_dates() -> None:
    returns = make_returns()
    labels = pd.Series(
        [0, 1],
        index=pd.to_datetime(["2030-01-01", "2030-01-02"]),
        name="regime",
    )

    with pytest.raises(RegimeCorrelationError, match="no overlapping dates"):
        calculate_regime_correlation_matrices(returns, labels)


def test_regime_correlation_rejects_duplicate_return_rows() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "ticker": ["SPY", "SPY"],
            "return": [0.01, 0.02],
        }
    )
    labels = make_regime_labels()

    with pytest.raises(RegimeCorrelationError, match="duplicate"):
        calculate_regime_correlation_matrices(returns, labels)


def test_regime_covariance_rejects_invalid_annualization_factor() -> None:
    returns = make_returns()
    labels = make_regime_labels()

    with pytest.raises(RegimeCorrelationError, match="Annualization factor"):
        calculate_regime_covariance_matrices(
            returns,
            labels,
            annualization_factor=0,
        )
