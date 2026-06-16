from collections.abc import Mapping, Sequence

import pandas as pd
from sklearn.preprocessing import StandardScaler

from regime_risk_engine.features.correlation import build_correlation_features
from regime_risk_engine.features.rolling import build_rolling_features
from regime_risk_engine.features.trend import build_trend_features


class FeatureMatrixError(ValueError):
    """Raised when the regime feature matrix cannot be built."""


def build_regime_feature_matrix(
    returns: pd.DataFrame,
    asset_feature_windows: Mapping[str, int] | None = None,
    correlation_windows: Mapping[str, int] | None = None,
    correlation_pairs: Sequence[tuple[str, str]] | None = None,
    scale_features: bool = True,
    drop_missing: bool = True,
    return_col: str = "return",
) -> pd.DataFrame:
    """Build a date-indexed regime feature matrix.

    Args:
        returns: Long-format return data with date, ticker, and return columns.
        asset_feature_windows: Windows for rolling and trend asset-level features.
        correlation_windows: Windows for date-level correlation features.
        correlation_pairs: Optional selected ticker pairs for pairwise correlations.
        scale_features: Whether to standardize feature columns.
        drop_missing: Whether to drop rows containing missing feature values.
        return_col: Name of the return column.

    Returns:
        Date-indexed feature matrix.
    """
    rolling_features = build_rolling_features(
        returns=returns,
        windows=asset_feature_windows,
        return_col=return_col,
    )
    trend_features = build_trend_features(
        returns=returns,
        windows=asset_feature_windows,
        return_col=return_col,
    )
    correlation_features = build_correlation_features(
        returns=returns,
        pairs=correlation_pairs,
        windows=correlation_windows,
        return_col=return_col,
    )

    asset_features = rolling_features.merge(
        trend_features,
        on=["date", "ticker"],
        how="inner",
    )

    asset_feature_matrix = pivot_asset_features_to_matrix(asset_features)
    correlation_feature_matrix = _prepare_date_level_features(correlation_features)

    feature_matrix = asset_feature_matrix.merge(
        correlation_feature_matrix,
        left_index=True,
        right_index=True,
        how="inner",
    )

    feature_matrix = feature_matrix.sort_index()
    validate_feature_matrix(feature_matrix)

    if drop_missing:
        feature_matrix = feature_matrix.dropna(axis=0, how="any")

    if feature_matrix.empty:
        raise FeatureMatrixError("Feature matrix is empty after processing")

    if scale_features:
        feature_matrix = scale_feature_matrix(feature_matrix)

    validate_feature_matrix(feature_matrix)

    return feature_matrix


def pivot_asset_features_to_matrix(asset_features: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format asset features into date-indexed wide format."""
    _validate_asset_features(asset_features)

    normalized_features = asset_features.copy()
    normalized_features["date"] = pd.to_datetime(normalized_features["date"])
    normalized_features["ticker"] = (
        normalized_features["ticker"].astype(str).str.upper().str.strip()
    )

    feature_columns = [
        column
        for column in normalized_features.columns
        if column not in {"date", "ticker"}
    ]

    if not feature_columns:
        raise FeatureMatrixError("Asset feature data contains no feature columns")

    matrix = normalized_features.pivot(
        index="date",
        columns="ticker",
        values=feature_columns,
    )

    matrix.columns = [
        f"{feature_name}__{ticker}" for feature_name, ticker in matrix.columns
    ]

    matrix = matrix.sort_index(axis=1).sort_index()
    validate_feature_matrix(matrix)

    return matrix


def scale_feature_matrix(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Standardize feature matrix columns to mean 0 and standard deviation 1."""
    validate_feature_matrix(feature_matrix)

    if feature_matrix.empty:
        raise FeatureMatrixError("Cannot scale an empty feature matrix")

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_matrix)

    return pd.DataFrame(
        scaled_values,
        index=feature_matrix.index,
        columns=feature_matrix.columns,
    )


def validate_feature_matrix(feature_matrix: pd.DataFrame) -> None:
    """Validate a date-indexed feature matrix."""
    if not isinstance(feature_matrix.index, pd.DatetimeIndex):
        raise FeatureMatrixError("Feature matrix index must be a DatetimeIndex")

    if feature_matrix.index.has_duplicates:
        raise FeatureMatrixError("Feature matrix index contains duplicate dates")

    if feature_matrix.columns.has_duplicates:
        raise FeatureMatrixError("Feature matrix contains duplicate columns")

    if feature_matrix.empty:
        raise FeatureMatrixError("Feature matrix is empty")


def _prepare_date_level_features(features: pd.DataFrame) -> pd.DataFrame:
    if "date" not in features.columns:
        raise FeatureMatrixError("Date-level features must contain a date column")

    normalized_features = features.copy()
    normalized_features["date"] = pd.to_datetime(normalized_features["date"])
    normalized_features = normalized_features.set_index("date").sort_index()

    if normalized_features.index.has_duplicates:
        raise FeatureMatrixError("Date-level features contain duplicate dates")

    if normalized_features.columns.has_duplicates:
        raise FeatureMatrixError("Date-level features contain duplicate columns")

    if normalized_features.empty:
        raise FeatureMatrixError("Date-level features are empty")

    return normalized_features


def _validate_asset_features(asset_features: pd.DataFrame) -> None:
    required_columns = {"date", "ticker"}
    missing_columns = required_columns.difference(asset_features.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise FeatureMatrixError(f"Missing required asset feature column(s): {missing}")

    if asset_features.empty:
        raise FeatureMatrixError("Asset feature data is empty")

    duplicate_rows = asset_features.duplicated(subset=["date", "ticker"])

    if duplicate_rows.any():
        raise FeatureMatrixError(
            "Asset feature data contains duplicate date/ticker rows"
        )
