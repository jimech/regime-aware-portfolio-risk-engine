from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.validation.walk_forward import WalkForwardRegimeValidation


class RegimeModelSelectionError(ValueError):
    """Raised when regime model selection summaries cannot be calculated."""


@dataclass(frozen=True, slots=True)
class RegimeModelSelectionReport:
    """Container for regime model selection outputs."""

    model_summary: pd.DataFrame
    split_summary: pd.DataFrame
    ranking: pd.DataFrame


REQUIRED_DIAGNOSTIC_COLUMNS = {
    "split_id",
    "sample",
    "observation_count",
    "regime_count",
    "transition_rate",
    "dominant_regime_share",
    "silhouette_score",
    "calinski_harabasz_score",
}

REQUIRED_SPLIT_COLUMNS = {
    "split_id",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "train_observation_count",
    "test_observation_count",
    "train_regime_count",
    "test_regime_count",
    "train_transition_rate",
    "test_transition_rate",
}

DEFAULT_RANKING_METRIC = "test_silhouette_score_mean"


def compare_walk_forward_validations(
    validations: Mapping[str, WalkForwardRegimeValidation],
    ranking_metric: str = DEFAULT_RANKING_METRIC,
    higher_is_better: bool = True,
) -> RegimeModelSelectionReport:
    """Compare walk-forward validation results across regime models."""
    cleaned_validations = _validate_validation_mapping(validations)

    model_summary_rows: list[dict[str, object]] = []
    split_frames: list[pd.DataFrame] = []

    for model_name, validation in cleaned_validations.items():
        model_summary_rows.append(
            _summarize_single_validation(
                model_name=model_name,
                validation=validation,
            )
        )

        split_frame = validation.split_summary.copy()
        split_frame.insert(0, "model", model_name)
        split_frames.append(split_frame)

    model_summary = (
        pd.DataFrame(model_summary_rows).sort_values("model").reset_index(drop=True)
    )
    split_summary = (
        pd.concat(split_frames, ignore_index=True)
        .sort_values(["model", "split_id"])
        .reset_index(drop=True)
    )

    ranking = rank_model_summaries(
        model_summary=model_summary,
        ranking_metric=ranking_metric,
        higher_is_better=higher_is_better,
    )

    return RegimeModelSelectionReport(
        model_summary=model_summary,
        split_summary=split_summary,
        ranking=ranking,
    )


def rank_model_summaries(
    model_summary: pd.DataFrame,
    ranking_metric: str,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Rank model summaries by one selected numeric metric."""
    _validate_model_summary_for_ranking(
        model_summary=model_summary,
        ranking_metric=ranking_metric,
    )

    clean_metric = str(ranking_metric).strip()

    ranked = model_summary.copy()
    ranked[clean_metric] = pd.to_numeric(ranked[clean_metric], errors="coerce")

    if ranked[clean_metric].isna().all():
        raise RegimeModelSelectionError(
            f"Ranking metric contains only missing values: {clean_metric}"
        )

    ranked = ranked.sort_values(
        by=clean_metric,
        ascending=not higher_is_better,
        na_position="last",
    ).reset_index(drop=True)

    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    return ranked[
        [
            "rank",
            "model",
            clean_metric,
            "split_count",
            "test_regime_count_mean",
            "test_transition_rate_mean",
            "test_dominant_regime_share_mean",
        ]
    ]


def _summarize_single_validation(
    model_name: str,
    validation: WalkForwardRegimeValidation,
) -> dict[str, object]:
    _validate_walk_forward_validation(validation)

    diagnostic_summary = validation.diagnostic_summary.copy()
    split_summary = validation.split_summary.copy()

    train_diagnostics = diagnostic_summary[diagnostic_summary["sample"] == "train"]
    test_diagnostics = diagnostic_summary[diagnostic_summary["sample"] == "test"]

    if train_diagnostics.empty:
        raise RegimeModelSelectionError(
            f"Validation for {model_name} has no train diagnostics"
        )

    if test_diagnostics.empty:
        raise RegimeModelSelectionError(
            f"Validation for {model_name} has no test diagnostics"
        )

    return {
        "model": model_name,
        "split_count": int(split_summary["split_id"].nunique()),
        "train_observation_count_mean": _mean_metric(
            train_diagnostics,
            "observation_count",
        ),
        "test_observation_count_mean": _mean_metric(
            test_diagnostics,
            "observation_count",
        ),
        "train_regime_count_mean": _mean_metric(
            train_diagnostics,
            "regime_count",
        ),
        "test_regime_count_mean": _mean_metric(
            test_diagnostics,
            "regime_count",
        ),
        "train_transition_rate_mean": _mean_metric(
            train_diagnostics,
            "transition_rate",
        ),
        "test_transition_rate_mean": _mean_metric(
            test_diagnostics,
            "transition_rate",
        ),
        "train_dominant_regime_share_mean": _mean_metric(
            train_diagnostics,
            "dominant_regime_share",
        ),
        "test_dominant_regime_share_mean": _mean_metric(
            test_diagnostics,
            "dominant_regime_share",
        ),
        "train_silhouette_score_mean": _mean_metric(
            train_diagnostics,
            "silhouette_score",
        ),
        "test_silhouette_score_mean": _mean_metric(
            test_diagnostics,
            "silhouette_score",
        ),
        "train_calinski_harabasz_score_mean": _mean_metric(
            train_diagnostics,
            "calinski_harabasz_score",
        ),
        "test_calinski_harabasz_score_mean": _mean_metric(
            test_diagnostics,
            "calinski_harabasz_score",
        ),
    }


def _mean_metric(data: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(data[column], errors="coerce")

    return float(values.mean(skipna=True))


def _validate_validation_mapping(
    validations: Mapping[str, WalkForwardRegimeValidation],
) -> dict[str, WalkForwardRegimeValidation]:
    if not validations:
        raise RegimeModelSelectionError(
            "At least one walk-forward validation result is required"
        )

    cleaned_validations: dict[str, WalkForwardRegimeValidation] = {}

    for model_name, validation in validations.items():
        clean_model_name = str(model_name).strip()

        if not clean_model_name:
            raise RegimeModelSelectionError("Model names must be non-empty")

        if clean_model_name in cleaned_validations:
            raise RegimeModelSelectionError(f"Duplicate model name: {clean_model_name}")

        _validate_walk_forward_validation(validation)

        cleaned_validations[clean_model_name] = validation

    return cleaned_validations


def _validate_walk_forward_validation(
    validation: WalkForwardRegimeValidation,
) -> None:
    if validation.label_frame.empty:
        raise RegimeModelSelectionError("Validation label frame cannot be empty")

    if validation.split_summary.empty:
        raise RegimeModelSelectionError("Validation split summary cannot be empty")

    if validation.diagnostic_summary.empty:
        raise RegimeModelSelectionError("Validation diagnostic summary cannot be empty")

    missing_diagnostic_columns = REQUIRED_DIAGNOSTIC_COLUMNS.difference(
        validation.diagnostic_summary.columns
    )

    if missing_diagnostic_columns:
        missing = ", ".join(sorted(missing_diagnostic_columns))
        raise RegimeModelSelectionError(
            f"Missing diagnostic summary column(s): {missing}"
        )

    missing_split_columns = REQUIRED_SPLIT_COLUMNS.difference(
        validation.split_summary.columns
    )

    if missing_split_columns:
        missing = ", ".join(sorted(missing_split_columns))
        raise RegimeModelSelectionError(f"Missing split summary column(s): {missing}")

    if validation.split_summary["split_id"].duplicated().any():
        raise RegimeModelSelectionError("Split summary contains duplicate split ids")


def _validate_model_summary_for_ranking(
    model_summary: pd.DataFrame,
    ranking_metric: str,
) -> None:
    clean_metric = str(ranking_metric).strip()

    if model_summary.empty:
        raise RegimeModelSelectionError("Model summary cannot be empty")

    if "model" not in model_summary.columns:
        raise RegimeModelSelectionError("Model summary must contain a model column")

    if not clean_metric:
        raise RegimeModelSelectionError("Ranking metric must be non-empty")

    if clean_metric not in model_summary.columns:
        raise RegimeModelSelectionError(f"Ranking metric not found: {clean_metric}")

    if model_summary["model"].duplicated().any():
        raise RegimeModelSelectionError("Model summary contains duplicate models")
