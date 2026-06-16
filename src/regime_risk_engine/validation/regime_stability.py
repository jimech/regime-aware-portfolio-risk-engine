from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class RegimeStabilityError(ValueError):
    """Raised when regime stability diagnostics cannot be calculated."""


@dataclass(frozen=True, slots=True)
class RegimeStabilityReport:
    """Container for regime stability diagnostics."""

    distribution_summary: pd.DataFrame
    transition_summary: pd.DataFrame
    agreement_summary: pd.DataFrame


def calculate_regime_distribution(regime_labels: pd.Series) -> pd.DataFrame:
    """Calculate regime observation counts and shares."""
    labels = _validate_regime_labels(regime_labels)

    counts = labels.value_counts().sort_index()
    total_count = int(len(labels))

    rows: list[dict[str, float | int]] = []

    for regime, count in counts.items():
        observation_count = int(count)
        rows.append(
            {
                "regime": int(regime),
                "observation_count": observation_count,
                "observation_share": float(observation_count / total_count),
            }
        )

    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def calculate_regime_transition_matrix(
    regime_labels: pd.Series,
    normalize: bool = True,
) -> pd.DataFrame:
    """Calculate regime transition matrix.

    Rows represent the previous regime.
    Columns represent the next regime.
    """
    labels = _validate_regime_labels(regime_labels)

    if len(labels) < 2:
        raise RegimeStabilityError(
            "At least two regime observations are required for transitions"
        )

    previous_labels = labels.iloc[:-1].to_numpy(dtype=int)
    next_labels = labels.iloc[1:].to_numpy(dtype=int)
    regimes = sorted(set(previous_labels.tolist()).union(next_labels.tolist()))

    matrix = pd.DataFrame(0.0, index=regimes, columns=regimes)

    for previous_regime, next_regime in zip(
        previous_labels,
        next_labels,
        strict=False,
    ):
        matrix.loc[int(previous_regime), int(next_regime)] += 1.0

    if normalize:
        row_sums = matrix.sum(axis=1).replace(0.0, np.nan)
        matrix = matrix.div(row_sums, axis=0).fillna(0.0)

    matrix.index.name = "from_regime"
    matrix.columns.name = "to_regime"

    return matrix


def calculate_regime_transition_summary(regime_labels: pd.Series) -> dict[str, float]:
    """Calculate transition-count and transition-rate diagnostics."""
    labels = _validate_regime_labels(regime_labels)

    if len(labels) < 2:
        raise RegimeStabilityError(
            "At least two regime observations are required for transitions"
        )

    transition_count = int(labels.ne(labels.shift()).iloc[1:].sum())
    possible_transition_count = len(labels) - 1
    transition_rate = transition_count / possible_transition_count
    regime_count = int(labels.nunique())
    dominant_regime_share = float(labels.value_counts(normalize=True).max())

    return {
        "observation_count": float(len(labels)),
        "regime_count": float(regime_count),
        "transition_count": float(transition_count),
        "transition_rate": float(transition_rate),
        "dominant_regime_share": dominant_regime_share,
    }


def calculate_label_agreement(
    reference_labels: pd.Series,
    candidate_labels: pd.Series,
) -> dict[str, float]:
    """Calculate agreement diagnostics between two regime label series.

    Adjusted Rand Index and normalized mutual information are label-invariant,
    which is important because unsupervised regime IDs are arbitrary.
    """
    reference = _validate_regime_labels(reference_labels)
    candidate = _validate_regime_labels(candidate_labels)

    overlapping_dates = reference.index.intersection(candidate.index).sort_values()

    if overlapping_dates.empty:
        raise RegimeStabilityError("Regime label series have no overlapping dates")

    aligned_reference = reference.loc[overlapping_dates]
    aligned_candidate = candidate.loc[overlapping_dates]

    reference_values = aligned_reference.to_numpy(dtype=int)
    candidate_values = aligned_candidate.to_numpy(dtype=int)

    direct_match_rate = float((reference_values == candidate_values).mean())
    adjusted_rand = float(adjusted_rand_score(reference_values, candidate_values))
    normalized_mutual_info = float(
        normalized_mutual_info_score(reference_values, candidate_values)
    )

    return {
        "observation_count": float(len(overlapping_dates)),
        "direct_match_rate": direct_match_rate,
        "adjusted_rand_index": adjusted_rand,
        "normalized_mutual_information": normalized_mutual_info,
    }


def build_regime_stability_report(
    regime_label_sets: Mapping[str, pd.Series],
    reference_name: str | None = None,
) -> RegimeStabilityReport:
    """Build regime distribution, transition, and agreement diagnostics."""
    cleaned_label_sets = _validate_label_set_mapping(regime_label_sets)

    distribution_frames: list[pd.DataFrame] = []
    transition_rows: list[dict[str, float | str]] = []

    for model_name, labels in cleaned_label_sets.items():
        distribution = calculate_regime_distribution(labels)
        distribution.insert(0, "model", model_name)
        distribution_frames.append(distribution)

        transition_summary = calculate_regime_transition_summary(labels)
        transition_rows.append({"model": model_name, **transition_summary})

    distribution_summary = pd.concat(distribution_frames, ignore_index=True)
    transition_summary_frame = pd.DataFrame(transition_rows)

    agreement_summary = _build_agreement_summary(
        regime_label_sets=cleaned_label_sets,
        reference_name=reference_name,
    )

    return RegimeStabilityReport(
        distribution_summary=distribution_summary,
        transition_summary=transition_summary_frame,
        agreement_summary=agreement_summary,
    )


def _build_agreement_summary(
    regime_label_sets: Mapping[str, pd.Series],
    reference_name: str | None,
) -> pd.DataFrame:
    model_names = list(regime_label_sets)

    columns = [
        "reference_model",
        "candidate_model",
        "observation_count",
        "direct_match_rate",
        "adjusted_rand_index",
        "normalized_mutual_information",
    ]

    if len(model_names) < 2:
        return pd.DataFrame(columns=columns)

    if reference_name is None:
        pairs = list(combinations(model_names, 2))
    else:
        clean_reference_name = str(reference_name).strip()

        if clean_reference_name not in regime_label_sets:
            raise RegimeStabilityError(
                f"Reference model not found: {clean_reference_name}"
            )

        pairs = [
            (clean_reference_name, model_name)
            for model_name in model_names
            if model_name != clean_reference_name
        ]

    rows: list[dict[str, float | str]] = []

    for reference_model, candidate_model in pairs:
        agreement = calculate_label_agreement(
            reference_labels=regime_label_sets[reference_model],
            candidate_labels=regime_label_sets[candidate_model],
        )
        rows.append(
            {
                "reference_model": reference_model,
                "candidate_model": candidate_model,
                **agreement,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _validate_label_set_mapping(
    regime_label_sets: Mapping[str, pd.Series],
) -> dict[str, pd.Series]:
    if not regime_label_sets:
        raise RegimeStabilityError("At least one regime label set is required")

    cleaned_label_sets: dict[str, pd.Series] = {}

    for model_name, labels in regime_label_sets.items():
        clean_model_name = str(model_name).strip()

        if not clean_model_name:
            raise RegimeStabilityError("Model names must be non-empty")

        if clean_model_name in cleaned_label_sets:
            raise RegimeStabilityError(f"Duplicate model name: {clean_model_name}")

        cleaned_label_sets[clean_model_name] = _validate_regime_labels(labels)

    return cleaned_label_sets


def _validate_regime_labels(regime_labels: pd.Series) -> pd.Series:
    if not isinstance(regime_labels.index, pd.DatetimeIndex):
        raise RegimeStabilityError("Regime labels index must be a DatetimeIndex")

    if regime_labels.empty:
        raise RegimeStabilityError("Regime labels cannot be empty")

    if regime_labels.index.has_duplicates:
        raise RegimeStabilityError("Regime labels contain duplicate dates")

    numeric_labels = pd.to_numeric(regime_labels, errors="coerce")

    if numeric_labels.isna().any():
        raise RegimeStabilityError(
            "Regime labels contain missing or non-numeric values"
        )

    integer_labels = numeric_labels.astype("int64")

    if not numeric_labels.eq(integer_labels).all():
        raise RegimeStabilityError("Regime labels must be integer-like values")

    labels = pd.Series(
        integer_labels,
        index=regime_labels.index,
        name="regime",
    )
    labels.index = pd.to_datetime(labels.index)

    return labels.sort_index()
