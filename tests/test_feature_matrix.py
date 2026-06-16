import pandas as pd
import pytest

from regime_risk_engine.features.matrix import (
    FeatureMatrixError,
    build_regime_feature_matrix,
    pivot_asset_features_to_matrix,
    scale_feature_matrix,
    validate_feature_matrix,
)


def make_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")

    return pd.DataFrame(
        {
            "date": list(dates) * 3,
            "ticker": ["SPY"] * 8 + ["TLT"] * 8 + ["GLD"] * 8,
            "return": [
                0.01,
                0.02,
                -0.01,
                0.03,
                0.01,
                -0.02,
                0.01,
                0.00,
                -0.01,
                -0.02,
                0.01,
                -0.01,
                0.00,
                0.01,
                -0.02,
                0.01,
                0.02,
                0.01,
                0.00,
                -0.01,
                0.02,
                0.01,
                -0.01,
                0.00,
            ],
        }
    )


def make_asset_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                ]
            ),
            "ticker": ["SPY", "TLT", "SPY", "TLT"],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_pivot_asset_features_to_matrix() -> None:
    asset_features = make_asset_features()

    matrix = pivot_asset_features_to_matrix(asset_features)

    assert isinstance(matrix.index, pd.DatetimeIndex)
    assert matrix.shape == (2, 4)
    assert set(matrix.columns) == {
        "feature_a__SPY",
        "feature_a__TLT",
        "feature_b__SPY",
        "feature_b__TLT",
    }
    assert matrix.loc[pd.Timestamp("2020-01-01"), "feature_a__SPY"] == 1.0


def test_pivot_asset_features_rejects_missing_columns() -> None:
    asset_features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "feature_a": [1.0],
        }
    )

    with pytest.raises(FeatureMatrixError, match="Missing required asset"):
        pivot_asset_features_to_matrix(asset_features)


def test_pivot_asset_features_rejects_duplicate_date_ticker_rows() -> None:
    asset_features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "ticker": ["SPY", "SPY"],
            "feature_a": [1.0, 2.0],
        }
    )

    with pytest.raises(FeatureMatrixError, match="duplicate"):
        pivot_asset_features_to_matrix(asset_features)


def test_scale_feature_matrix_standardizes_columns() -> None:
    matrix = pd.DataFrame(
        {
            "feature_a__SPY": [1.0, 2.0, 3.0],
            "feature_a__TLT": [10.0, 20.0, 30.0],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )

    scaled = scale_feature_matrix(matrix)

    assert scaled.mean().abs().max() < 1e-12


def test_validate_feature_matrix_rejects_non_datetime_index() -> None:
    matrix = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
        },
        index=[0, 1],
    )

    with pytest.raises(FeatureMatrixError, match="DatetimeIndex"):
        validate_feature_matrix(matrix)


def test_validate_feature_matrix_rejects_duplicate_index() -> None:
    matrix = pd.DataFrame(
        {
            "feature": [1.0, 2.0],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-01"]),
    )

    with pytest.raises(FeatureMatrixError, match="duplicate dates"):
        validate_feature_matrix(matrix)


def test_build_regime_feature_matrix() -> None:
    returns = make_returns()

    matrix = build_regime_feature_matrix(
        returns,
        asset_feature_windows={"three_day": 3},
        correlation_windows={"three_day": 3},
        correlation_pairs=[("SPY", "TLT")],
        scale_features=False,
    )

    assert isinstance(matrix.index, pd.DatetimeIndex)
    assert len(matrix) > 0
    assert "avg_pairwise_corr_three_day_3d" in matrix.columns
    assert "rolling_corr_SPY_TLT_three_day_3d" in matrix.columns
    assert any(column.startswith("rolling_return_three_day_3d__") for column in matrix)
    assert any(column.startswith("momentum_three_day_3d__") for column in matrix)
    assert not matrix.isna().any().any()


def test_build_regime_feature_matrix_with_scaling() -> None:
    returns = make_returns()

    matrix = build_regime_feature_matrix(
        returns,
        asset_feature_windows={"three_day": 3},
        correlation_windows={"three_day": 3},
        correlation_pairs=[("SPY", "TLT")],
        scale_features=True,
    )

    assert isinstance(matrix.index, pd.DatetimeIndex)
    assert len(matrix) > 0
    assert matrix.mean().abs().max() < 1e-12


def test_build_regime_feature_matrix_can_keep_missing_values() -> None:
    returns = make_returns()

    matrix = build_regime_feature_matrix(
        returns,
        asset_feature_windows={"three_day": 3},
        correlation_windows={"three_day": 3},
        correlation_pairs=[("SPY", "TLT")],
        scale_features=False,
        drop_missing=False,
    )

    assert matrix.isna().any().any()
