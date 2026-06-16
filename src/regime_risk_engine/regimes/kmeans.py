from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans

from regime_risk_engine.regimes.base import (
    RegimeDetectionResult,
    RegimeModelError,
    validate_regime_feature_matrix,
)


@dataclass(frozen=True, slots=True)
class KMeansRegimeConfig:
    """Configuration for K-Means regime detection."""

    n_regimes: int = 4
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 300


class KMeansRegimeModel:
    """K-Means baseline model for market regime detection."""

    def __init__(self, config: KMeansRegimeConfig | None = None) -> None:
        self.config = config or KMeansRegimeConfig()
        self._validate_config()

        self._model: KMeans | None = None
        self._feature_columns: list[str] | None = None

    def fit(self, feature_matrix: pd.DataFrame) -> "KMeansRegimeModel":
        """Fit the K-Means regime model."""
        validate_regime_feature_matrix(feature_matrix)

        model = KMeans(
            n_clusters=self.config.n_regimes,
            random_state=self.config.random_state,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
        )
        model.fit(feature_matrix)

        self._model = model
        self._feature_columns = list(feature_matrix.columns)

        return self

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """Predict regime labels for a feature matrix."""
        validate_regime_feature_matrix(feature_matrix)
        self._validate_is_fitted()
        self._validate_feature_columns(feature_matrix)

        model = self._model

        if model is None:
            raise RegimeModelError("K-Means regime model has not been fit")

        labels = model.predict(feature_matrix)

        return pd.Series(
            labels,
            index=feature_matrix.index,
            name="regime",
            dtype="int64",
        )

    def fit_predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """Fit the model and return regime labels."""
        self.fit(feature_matrix)

        return self.predict(feature_matrix)

    def fit_predict_result(
        self,
        feature_matrix: pd.DataFrame,
    ) -> RegimeDetectionResult:
        """Fit the model and return a structured regime detection result."""
        labels = self.fit_predict(feature_matrix)

        return RegimeDetectionResult(
            labels=labels,
            model_name="kmeans",
            metadata={
                "n_regimes": self.config.n_regimes,
                "random_state": self.config.random_state,
                "n_features": feature_matrix.shape[1],
                "n_observations": feature_matrix.shape[0],
            },
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fit."""
        return self._model is not None

    @property
    def feature_columns(self) -> list[str]:
        """Return feature columns used during fitting."""
        if self._feature_columns is None:
            return []

        return self._feature_columns.copy()

    def _validate_config(self) -> None:
        if self.config.n_regimes <= 1:
            raise RegimeModelError("Number of regimes must be greater than 1")

        if self.config.n_init <= 0:
            raise RegimeModelError("n_init must be positive")

        if self.config.max_iter <= 0:
            raise RegimeModelError("max_iter must be positive")

    def _validate_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RegimeModelError("K-Means regime model has not been fit")

    def _validate_feature_columns(self, feature_matrix: pd.DataFrame) -> None:
        if self._feature_columns is None:
            raise RegimeModelError("K-Means regime model has not been fit")

        incoming_columns = list(feature_matrix.columns)

        if incoming_columns != self._feature_columns:
            raise RegimeModelError(
                "Feature columns must match columns used during fitting"
            )
