from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, Self

import pandas as pd
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from regime_risk_engine.validation.splits import (
    TimeSeriesSplit,
    validate_time_series_split,
)


class WalkForwardValidationError(ValueError):
    """Raised when walk-forward regime validation cannot be calculated."""


class RegimeModelLike(Protocol):
    """Protocol for regime models used in walk-forward validation."""

    def fit(self, feature_matrix: pd.DataFrame) -> Self:
        """Fit model on feature matrix."""

    def predict(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """Predict regime labels for feature matrix."""


@dataclass(frozen=True, slots=True)
class WalkForwardRegimeValidation:
    """Container for walk-forward regime validation output."""

    label_frame: pd.DataFrame
    split_summary: pd.DataFrame
    diagnostic_summary: pd.DataFrame


def run_walk_forward_regime_validation(
    feature_matrix: pd.DataFrame,
    splits: Sequence[TimeSeriesSplit],
    model_factory: Callable[[], RegimeModelLike],
) -> WalkForwardRegimeValidation:
    """Run walk-forward validation for a regime detection model.

    Args:
        feature_matrix: Date-indexed feature matrix.
        splits: Chronological train/test splits.
        model_factory: Callable that creates a fresh unfitted model instance.

    Returns:
        WalkForwardRegimeValidation output.
    """
    clean_feature_matrix = _validate_feature_matrix(feature_matrix)
    clean_splits = _validate_splits(splits)
    _validate_feature_matrix_covers_splits(clean_feature_matrix, clean_splits)

    label_frames: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for split in clean_splits:
        train_features, test_features = _split_feature_matrix(
            feature_matrix=clean_feature_matrix,
            split=split,
        )

        model = model_factory()
        model.fit(train_features)

        train_labels = _validate_predicted_labels(
            labels=model.predict(train_features),
            expected_dates=pd.DatetimeIndex(train_features.index),
            sample_name="train",
            split_id=split.split_id,
        )
        test_labels = _validate_predicted_labels(
            labels=model.predict(test_features),
            expected_dates=pd.DatetimeIndex(test_features.index),
            sample_name="test",
            split_id=split.split_id,
        )

        label_frames.append(
            _build_label_frame_for_split(
                split_id=split.split_id,
                train_labels=train_labels,
                test_labels=test_labels,
            )
        )

        split_rows.append(
            _build_split_summary_row(
                split=split,
                train_labels=train_labels,
                test_labels=test_labels,
            )
        )

        diagnostic_rows.append(
            _build_diagnostic_row(
                split_id=split.split_id,
                sample="train",
                features=train_features,
                labels=train_labels,
            )
        )
        diagnostic_rows.append(
            _build_diagnostic_row(
                split_id=split.split_id,
                sample="test",
                features=test_features,
                labels=test_labels,
            )
        )

    label_frame = pd.concat(label_frames, ignore_index=True)
    split_summary = (
        pd.DataFrame(split_rows).sort_values("split_id").reset_index(drop=True)
    )
    diagnostic_summary = (
        pd.DataFrame(diagnostic_rows)
        .sort_values(["split_id", "sample"])
        .reset_index(drop=True)
    )

    return WalkForwardRegimeValidation(
        label_frame=label_frame,
        split_summary=split_summary,
        diagnostic_summary=diagnostic_summary,
    )


def _validate_feature_matrix(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_matrix.index, pd.DatetimeIndex):
        raise WalkForwardValidationError("Feature matrix index must be a DatetimeIndex")

    if feature_matrix.empty:
        raise WalkForwardValidationError("Feature matrix cannot be empty")

    if feature_matrix.index.has_duplicates:
        raise WalkForwardValidationError("Feature matrix contains duplicate dates")

    if feature_matrix.columns.has_duplicates:
        raise WalkForwardValidationError("Feature matrix contains duplicate columns")

    if feature_matrix.isna().any().any():
        raise WalkForwardValidationError("Feature matrix contains missing values")

    clean_feature_matrix = feature_matrix.copy().sort_index()

    for column in clean_feature_matrix.columns:
        clean_feature_matrix[column] = pd.to_numeric(
            clean_feature_matrix[column],
            errors="coerce",
        )

    if clean_feature_matrix.isna().any().any():
        raise WalkForwardValidationError("Feature matrix contains non-numeric values")

    return clean_feature_matrix


def _validate_splits(splits: Sequence[TimeSeriesSplit]) -> list[TimeSeriesSplit]:
    if not splits:
        raise WalkForwardValidationError("At least one validation split is required")

    clean_splits = list(splits)
    seen_split_ids: set[int] = set()

    for split in clean_splits:
        validate_time_series_split(split)

        if split.split_id in seen_split_ids:
            raise WalkForwardValidationError(
                f"Duplicate validation split id: {split.split_id}"
            )

        seen_split_ids.add(split.split_id)

    return sorted(clean_splits, key=lambda split: split.split_id)


def _validate_feature_matrix_covers_splits(
    feature_matrix: pd.DataFrame,
    splits: Sequence[TimeSeriesSplit],
) -> None:
    last_test_end = max(split.test_end for split in splits)
    feature_end = pd.Timestamp(feature_matrix.index.max())

    if feature_end <= last_test_end:
        raise WalkForwardValidationError(
            "Feature matrix is missing test dates after the final validation split"
        )


def _split_feature_matrix(
    feature_matrix: pd.DataFrame,
    split: TimeSeriesSplit,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_train_dates = split.train_dates.difference(feature_matrix.index)
    missing_test_dates = split.test_dates.difference(feature_matrix.index)

    if not missing_train_dates.empty:
        raise WalkForwardValidationError(
            f"Feature matrix is missing train dates for split {split.split_id}"
        )

    if not missing_test_dates.empty:
        raise WalkForwardValidationError(
            f"Feature matrix is missing test dates for split {split.split_id}"
        )

    train_features = feature_matrix.loc[split.train_dates].copy()
    test_features = feature_matrix.loc[split.test_dates].copy()

    if train_features.empty:
        raise WalkForwardValidationError(
            f"Train feature matrix is empty for split {split.split_id}"
        )

    if test_features.empty:
        raise WalkForwardValidationError(
            f"Test feature matrix is empty for split {split.split_id}"
        )

    return train_features, test_features


def _validate_predicted_labels(
    labels: pd.Series,
    expected_dates: pd.DatetimeIndex,
    sample_name: str,
    split_id: int,
) -> pd.Series:
    if not isinstance(labels, pd.Series):
        raise WalkForwardValidationError(
            f"Predicted {sample_name} labels for split {split_id} must be a Series"
        )

    if not isinstance(labels.index, pd.DatetimeIndex):
        raise WalkForwardValidationError(
            f"Predicted {sample_name} labels for split {split_id} "
            "must use a DatetimeIndex"
        )

    if labels.empty:
        raise WalkForwardValidationError(
            f"Predicted {sample_name} labels for split {split_id} cannot be empty"
        )

    if labels.index.has_duplicates:
        raise WalkForwardValidationError(
            f"Predicted {sample_name} labels for split {split_id} "
            "contain duplicate dates"
        )

    if not labels.index.equals(expected_dates):
        raise WalkForwardValidationError(
            f"Predicted {sample_name} labels for split {split_id} "
            "must match feature matrix dates"
        )

    numeric_labels = pd.to_numeric(labels, errors="coerce")

    if numeric_labels.isna().any():
        raise WalkForwardValidationError(
            f"Predicted {sample_name} labels for split {split_id} "
            "contain missing or non-numeric values"
        )

    integer_labels = numeric_labels.astype("int64")

    if not numeric_labels.eq(integer_labels).all():
        raise WalkForwardValidationError(
            f"Predicted {sample_name} labels for split {split_id} "
            "must be integer-like values"
        )

    return pd.Series(
        integer_labels,
        index=labels.index,
        name="regime",
    )


def _build_label_frame_for_split(
    split_id: int,
    train_labels: pd.Series,
    test_labels: pd.Series,
) -> pd.DataFrame:
    train_frame = pd.DataFrame(
        {
            "split_id": split_id,
            "sample": "train",
            "date": train_labels.index,
            "regime": train_labels.to_numpy(dtype=int),
        }
    )
    test_frame = pd.DataFrame(
        {
            "split_id": split_id,
            "sample": "test",
            "date": test_labels.index,
            "regime": test_labels.to_numpy(dtype=int),
        }
    )

    return pd.concat([train_frame, test_frame], ignore_index=True)


def _build_split_summary_row(
    split: TimeSeriesSplit,
    train_labels: pd.Series,
    test_labels: pd.Series,
) -> dict[str, object]:
    return {
        "split_id": int(split.split_id),
        "train_start": split.train_start,
        "train_end": split.train_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
        "train_observation_count": int(len(train_labels)),
        "test_observation_count": int(len(test_labels)),
        "train_regime_count": int(train_labels.nunique()),
        "test_regime_count": int(test_labels.nunique()),
        "train_transition_rate": _calculate_transition_rate(train_labels),
        "test_transition_rate": _calculate_transition_rate(test_labels),
    }


def _build_diagnostic_row(
    split_id: int,
    sample: str,
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict[str, object]:
    return {
        "split_id": int(split_id),
        "sample": sample,
        "observation_count": int(len(labels)),
        "regime_count": int(labels.nunique()),
        "transition_rate": _calculate_transition_rate(labels),
        "dominant_regime_share": float(labels.value_counts(normalize=True).max()),
        "silhouette_score": _calculate_silhouette_score(features, labels),
        "calinski_harabasz_score": _calculate_calinski_harabasz_score(
            features,
            labels,
        ),
    }


def _calculate_transition_rate(labels: pd.Series) -> float:
    if len(labels) < 2:
        return float("nan")

    transition_count = int(labels.ne(labels.shift()).iloc[1:].sum())
    possible_transition_count = len(labels) - 1

    return float(transition_count / possible_transition_count)


def _calculate_silhouette_score(
    features: pd.DataFrame,
    labels: pd.Series,
) -> float:
    unique_label_count = int(labels.nunique())
    observation_count = int(len(labels))

    if unique_label_count < 2 or unique_label_count >= observation_count:
        return float("nan")

    return float(silhouette_score(features.to_numpy(), labels.to_numpy(dtype=int)))


def _calculate_calinski_harabasz_score(
    features: pd.DataFrame,
    labels: pd.Series,
) -> float:
    unique_label_count = int(labels.nunique())
    observation_count = int(len(labels))

    if unique_label_count < 2 or unique_label_count >= observation_count:
        return float("nan")

    return float(
        calinski_harabasz_score(
            features.to_numpy(),
            labels.to_numpy(dtype=int),
        )
    )
