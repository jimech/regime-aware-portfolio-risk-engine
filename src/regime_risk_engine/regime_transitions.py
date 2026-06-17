from dataclasses import dataclass

import pandas as pd


class RegimeTransitionError(ValueError):
    """Raised when regime transition analysis cannot be completed."""


@dataclass(frozen=True, slots=True)
class RegimeTransitionSummary:
    """Summary of regime transition behavior."""

    transition_counts: pd.DataFrame
    transition_probabilities: pd.DataFrame
    regime_persistence: pd.DataFrame
    regime_durations: pd.DataFrame
    most_likely_next_regime: pd.DataFrame
    most_persistent_regime: int | None
    least_persistent_regime: int | None
    narrative: str


def build_regime_transition_summary(
    regime_labels: pd.Series,
) -> RegimeTransitionSummary:
    """Analyze regime transition probabilities and persistence."""
    clean_labels = _validate_and_prepare_regime_labels(regime_labels)

    regimes = sorted(int(regime) for regime in clean_labels.unique())

    transition_counts = _build_transition_count_matrix(
        regime_labels=clean_labels,
        regimes=regimes,
    )
    transition_probabilities = _build_transition_probability_matrix(
        transition_counts=transition_counts,
    )
    regime_persistence = _build_regime_persistence_table(
        transition_probabilities=transition_probabilities,
        transition_counts=transition_counts,
    )
    regime_durations = _build_regime_duration_table(clean_labels)
    most_likely_next_regime = _build_most_likely_next_regime_table(
        transition_probabilities=transition_probabilities,
    )

    most_persistent_regime = _extract_persistence_extreme(
        regime_persistence=regime_persistence,
        ascending=False,
    )
    least_persistent_regime = _extract_persistence_extreme(
        regime_persistence=regime_persistence,
        ascending=True,
    )

    narrative = _build_narrative(
        regime_count=len(regimes),
        transition_count=int(transition_counts.to_numpy().sum()),
        most_persistent_regime=most_persistent_regime,
        least_persistent_regime=least_persistent_regime,
        regime_persistence=regime_persistence,
    )

    return RegimeTransitionSummary(
        transition_counts=transition_counts,
        transition_probabilities=transition_probabilities,
        regime_persistence=regime_persistence,
        regime_durations=regime_durations,
        most_likely_next_regime=most_likely_next_regime,
        most_persistent_regime=most_persistent_regime,
        least_persistent_regime=least_persistent_regime,
        narrative=narrative,
    )


def _validate_and_prepare_regime_labels(regime_labels: pd.Series) -> pd.Series:
    if regime_labels.empty:
        raise RegimeTransitionError("Regime labels cannot be empty")

    if not isinstance(regime_labels.index, pd.DatetimeIndex):
        raise RegimeTransitionError("Regime labels index must be a DatetimeIndex")

    if regime_labels.isna().any():
        raise RegimeTransitionError("Regime labels cannot contain missing values")

    numeric_labels = pd.to_numeric(regime_labels, errors="coerce")

    if numeric_labels.isna().any():
        raise RegimeTransitionError("Regime labels must be numeric")

    clean_labels = numeric_labels.astype(int).sort_index()

    if len(clean_labels) < 2:
        raise RegimeTransitionError("At least two regime observations are required")

    if clean_labels.nunique() < 2:
        raise RegimeTransitionError("At least two unique regimes are required")

    return pd.Series(
        clean_labels.to_numpy(dtype=int),
        index=clean_labels.index,
        name="regime",
    )


def _build_transition_count_matrix(
    regime_labels: pd.Series,
    regimes: list[int],
) -> pd.DataFrame:
    counts = pd.DataFrame(
        0,
        index=pd.Index(regimes, name="current_regime"),
        columns=pd.Index(regimes, name="next_regime"),
        dtype=int,
    )

    current_values = regime_labels.iloc[:-1].to_numpy(dtype=int)
    next_values = regime_labels.iloc[1:].to_numpy(dtype=int)

    for current_regime, next_regime in zip(
        current_values,
        next_values,
        strict=True,
    ):
        counts.loc[int(current_regime), int(next_regime)] += 1

    return counts


def _build_transition_probability_matrix(
    transition_counts: pd.DataFrame,
) -> pd.DataFrame:
    row_sums = transition_counts.sum(axis=1)
    probabilities = transition_counts.div(row_sums.replace(0, pd.NA), axis=0)
    probabilities = probabilities.fillna(0.0)

    return probabilities.astype(float)


def _build_regime_persistence_table(
    transition_probabilities: pd.DataFrame,
    transition_counts: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for regime in transition_probabilities.index:
        persistence_probability = _to_float(
            transition_probabilities.loc[regime, regime]
        )
        transition_observation_count = int(transition_counts.loc[regime].sum())

        rows.append(
            {
                "regime": int(regime),
                "persistence_probability": persistence_probability,
                "transition_observation_count": transition_observation_count,
                "expected_duration": _calculate_expected_duration(
                    persistence_probability
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def _calculate_expected_duration(persistence_probability: float) -> float:
    if persistence_probability >= 1.0:
        return float("inf")

    if persistence_probability <= 0.0:
        return 1.0

    return float(1.0 / (1.0 - persistence_probability))


def _build_regime_duration_table(regime_labels: pd.Series) -> pd.DataFrame:
    rows = []

    current_regime = int(regime_labels.iloc[0])
    start_date = pd.Timestamp(regime_labels.index[0])
    previous_date = start_date
    duration = 1

    for date, regime_value in regime_labels.iloc[1:].items():
        regime = int(regime_value)
        timestamp = pd.Timestamp(date)

        if regime == current_regime:
            duration += 1
            previous_date = timestamp
            continue

        rows.append(
            {
                "regime": current_regime,
                "start_date": start_date.date().isoformat(),
                "end_date": previous_date.date().isoformat(),
                "duration": duration,
            }
        )

        current_regime = regime
        start_date = timestamp
        previous_date = timestamp
        duration = 1

    rows.append(
        {
            "regime": current_regime,
            "start_date": start_date.date().isoformat(),
            "end_date": previous_date.date().isoformat(),
            "duration": duration,
        }
    )

    return pd.DataFrame(rows)


def _build_most_likely_next_regime_table(
    transition_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for regime in transition_probabilities.index:
        row = transition_probabilities.loc[regime]
        next_regime = int(row.idxmax())
        probability = _to_float(row.loc[next_regime])

        rows.append(
            {
                "current_regime": int(regime),
                "most_likely_next_regime": next_regime,
                "transition_probability": probability,
            }
        )

    return pd.DataFrame(rows).sort_values("current_regime").reset_index(drop=True)


def _extract_persistence_extreme(
    regime_persistence: pd.DataFrame,
    ascending: bool,
) -> int | None:
    if regime_persistence.empty:
        return None

    sorted_table = regime_persistence.sort_values(
        "persistence_probability",
        ascending=ascending,
    )

    return int(sorted_table.iloc[0]["regime"])


def _build_narrative(
    regime_count: int,
    transition_count: int,
    most_persistent_regime: int | None,
    least_persistent_regime: int | None,
    regime_persistence: pd.DataFrame,
) -> str:
    narrative = (
        f"Regime transition analysis estimated transition probabilities across "
        f"{regime_count} regimes using {transition_count} observed transitions."
    )

    if most_persistent_regime is not None:
        probability = _lookup_persistence_probability(
            regime_persistence=regime_persistence,
            regime=most_persistent_regime,
        )
        narrative += (
            f" Regime {most_persistent_regime} was the most persistent regime "
            f"with a self-transition probability of {probability:.2%}."
        )

    if least_persistent_regime is not None:
        probability = _lookup_persistence_probability(
            regime_persistence=regime_persistence,
            regime=least_persistent_regime,
        )
        narrative += (
            f" Regime {least_persistent_regime} was the least persistent regime "
            f"with a self-transition probability of {probability:.2%}."
        )

    return narrative


def _lookup_persistence_probability(
    regime_persistence: pd.DataFrame,
    regime: int,
) -> float:
    row = regime_persistence[regime_persistence["regime"] == regime]

    if row.empty:
        return 0.0

    return _to_float(row.iloc[0]["persistence_probability"])


def _to_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise RegimeTransitionError("Expected numeric regime transition value")

    return float(numeric)
