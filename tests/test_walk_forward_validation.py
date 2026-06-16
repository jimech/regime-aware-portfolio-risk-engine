import pandas as pd
import pytest

from regime_risk_engine.validation.splits import (
    TimeSeriesSplit,
    build_expanding_window_splits,
)
from regime_risk_engine.validation.walk_forward import (
    WalkForwardRegimeValidation,
    WalkForwardValidationError,
    run_walk_forward_regime_validation,
)


class ThresholdRegimeModel:
    """Simple deterministic test model."""

    threshold_: float | None

    def __init__(self) -> None:
        self.threshold_ = None

    def fit(self, feature_matrix: pd.DataFrame) -> "ThresholdRegimeModel":
        self.threshold_ = float(feature_matrix["feature_1"].median())

        return self

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        if self.threshold_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        labels = (feature_matrix["feature_1"] > self.threshold_).astype(int)

        return pd.Series(
            labels,
            index=feature_matrix.index,
            name="regime",
        )


class BadIndexRegimeModel:
    def fit(self, feature_matrix: pd.DataFrame) -> "BadIndexRegimeModel":
        return self

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [0 for _ in range(len(feature_matrix))],
            index=range(len(feature_matrix)),
            name="regime",
        )


class MissingLabelRegimeModel:
    def fit(self, feature_matrix: pd.DataFrame) -> "MissingLabelRegimeModel":
        return self

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        labels = pd.Series(
            [0 for _ in range(len(feature_matrix))],
            index=feature_matrix.index,
            name="regime",
        )
        labels.iloc[0] = None

        return labels


def make_feature_matrix() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=12, freq="D")

    return pd.DataFrame(
        {
            "feature_1": [
                0.1,
                0.2,
                0.3,
                0.4,
                1.0,
                1.1,
                1.2,
                1.3,
                0.2,
                0.3,
                1.4,
                1.5,
            ],
            "feature_2": [
                1.0,
                1.1,
                1.2,
                1.3,
                0.4,
                0.3,
                0.2,
                0.1,
                1.4,
                1.5,
                0.0,
                -0.1,
            ],
        },
        index=dates,
    )


def make_splits() -> list[TimeSeriesSplit]:
    return build_expanding_window_splits(
        dates=pd.DatetimeIndex(make_feature_matrix().index),
        initial_train_size=5,
        test_size=3,
        max_splits=2,
    )


def test_run_walk_forward_regime_validation() -> None:
    result = run_walk_forward_regime_validation(
        feature_matrix=make_feature_matrix(),
        splits=make_splits(),
        model_factory=ThresholdRegimeModel,
    )

    assert isinstance(result, WalkForwardRegimeValidation)
    assert set(result.label_frame.columns) == {
        "split_id",
        "sample",
        "date",
        "regime",
    }
    assert set(result.label_frame["sample"]) == {"train", "test"}
    assert result.split_summary["split_id"].tolist() == [0, 1]
    assert len(result.diagnostic_summary) == 4
    assert set(result.diagnostic_summary["sample"]) == {"train", "test"}


def test_walk_forward_uses_each_split_train_and_test_dates() -> None:
    result = run_walk_forward_regime_validation(
        feature_matrix=make_feature_matrix(),
        splits=make_splits(),
        model_factory=ThresholdRegimeModel,
    )

    split_0 = make_splits()[0]
    split_0_frame = result.label_frame[result.label_frame["split_id"] == 0]

    train_dates = split_0_frame[split_0_frame["sample"] == "train"]["date"]
    test_dates = split_0_frame[split_0_frame["sample"] == "test"]["date"]

    assert list(train_dates) == list(split_0.train_dates)
    assert list(test_dates) == list(split_0.test_dates)


def test_walk_forward_split_summary_contains_regime_diagnostics() -> None:
    result = run_walk_forward_regime_validation(
        feature_matrix=make_feature_matrix(),
        splits=make_splits(),
        model_factory=ThresholdRegimeModel,
    )

    assert "train_regime_count" in result.split_summary.columns
    assert "test_regime_count" in result.split_summary.columns
    assert "train_transition_rate" in result.split_summary.columns
    assert "test_transition_rate" in result.split_summary.columns


def test_walk_forward_diagnostic_summary_contains_clustering_metrics() -> None:
    result = run_walk_forward_regime_validation(
        feature_matrix=make_feature_matrix(),
        splits=make_splits(),
        model_factory=ThresholdRegimeModel,
    )

    assert "silhouette_score" in result.diagnostic_summary.columns
    assert "calinski_harabasz_score" in result.diagnostic_summary.columns


def test_walk_forward_rejects_empty_splits() -> None:
    with pytest.raises(WalkForwardValidationError, match="At least one"):
        run_walk_forward_regime_validation(
            feature_matrix=make_feature_matrix(),
            splits=[],
            model_factory=ThresholdRegimeModel,
        )


def test_walk_forward_rejects_non_datetime_feature_index() -> None:
    feature_matrix = make_feature_matrix()
    feature_matrix.index = list(range(len(feature_matrix)))

    with pytest.raises(WalkForwardValidationError, match="DatetimeIndex"):
        run_walk_forward_regime_validation(
            feature_matrix=feature_matrix,
            splits=make_splits(),
            model_factory=ThresholdRegimeModel,
        )


def test_walk_forward_rejects_missing_feature_values() -> None:
    feature_matrix = make_feature_matrix()
    feature_matrix.iloc[0, 0] = None

    with pytest.raises(WalkForwardValidationError, match="missing values"):
        run_walk_forward_regime_validation(
            feature_matrix=feature_matrix,
            splits=make_splits(),
            model_factory=ThresholdRegimeModel,
        )


def test_walk_forward_rejects_missing_split_dates() -> None:
    feature_matrix = make_feature_matrix().iloc[:-1]
    splits = make_splits()

    with pytest.raises(WalkForwardValidationError, match="missing test dates"):
        run_walk_forward_regime_validation(
            feature_matrix=feature_matrix,
            splits=splits,
            model_factory=ThresholdRegimeModel,
        )


def test_walk_forward_rejects_bad_prediction_index() -> None:
    with pytest.raises(WalkForwardValidationError, match="DatetimeIndex"):
        run_walk_forward_regime_validation(
            feature_matrix=make_feature_matrix(),
            splits=make_splits(),
            model_factory=BadIndexRegimeModel,
        )


def test_walk_forward_rejects_missing_predicted_labels() -> None:
    with pytest.raises(WalkForwardValidationError, match="missing"):
        run_walk_forward_regime_validation(
            feature_matrix=make_feature_matrix(),
            splits=make_splits(),
            model_factory=MissingLabelRegimeModel,
        )
