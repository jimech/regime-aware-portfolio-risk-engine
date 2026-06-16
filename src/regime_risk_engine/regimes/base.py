from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


class RegimeModelError(ValueError):
    """Raised when a regime model cannot be fit or used."""


@dataclass(frozen=True, slots=True)
class RegimeDetectionResult:
    """Container for regime detection output."""

    labels: pd.Series
    model_name: str
    metadata: dict[str, object] = field(default_factory=dict)


class RegimeModel(Protocol):
    """Protocol for regime detection models."""

    def fit(self, feature_matrix: pd.DataFrame) -> "RegimeModel":
        """Fit a regime model."""

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """Predict regime labels."""

    def fit_predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """Fit a regime model and return labels."""


def validate_regime_feature_matrix(feature_matrix: pd.DataFrame) -> None:
    """Validate a date-indexed feature matrix for regime modeling."""
    if not isinstance(feature_matrix.index, pd.DatetimeIndex):
        raise RegimeModelError("Feature matrix index must be a DatetimeIndex")

    if feature_matrix.empty:
        raise RegimeModelError("Feature matrix is empty")

    if feature_matrix.index.has_duplicates:
        raise RegimeModelError("Feature matrix index contains duplicate dates")

    if feature_matrix.columns.has_duplicates:
        raise RegimeModelError("Feature matrix contains duplicate columns")

    if feature_matrix.isna().any().any():
        raise RegimeModelError("Feature matrix contains missing values")

    numeric_feature_count = feature_matrix.select_dtypes(include="number").shape[1]

    if numeric_feature_count != feature_matrix.shape[1]:
        raise RegimeModelError("Feature matrix must contain only numeric columns")
