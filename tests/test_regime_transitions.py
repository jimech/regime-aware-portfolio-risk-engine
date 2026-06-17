import pandas as pd
import pytest

from regime_risk_engine.research.regime_transitions import (
    RegimeTransitionError,
    RegimeTransitionSummary,
    build_regime_transition_summary,
)


def make_regime_labels() -> pd.Series:
    dates = pd.date_range("2020-01-01", periods=12, freq="D")

    return pd.Series(
        [0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1, 0],
        index=dates,
        name="regime",
    )


def test_build_regime_transition_summary() -> None:
    summary = build_regime_transition_summary(make_regime_labels())

    assert isinstance(summary, RegimeTransitionSummary)
    assert summary.transition_counts.shape == (3, 3)
    assert summary.transition_probabilities.shape == (3, 3)
    assert len(summary.regime_persistence) == 3
    assert len(summary.regime_durations) == 5
    assert len(summary.most_likely_next_regime) == 3
    assert summary.most_persistent_regime == 2
    assert summary.least_persistent_regime == 1
    assert "transition probabilities across 3 regimes" in summary.narrative


def test_transition_count_matrix_counts_observed_switches() -> None:
    summary = build_regime_transition_summary(make_regime_labels())

    counts = summary.transition_counts

    assert counts.loc[0, 0] == 2
    assert counts.loc[0, 1] == 1
    assert counts.loc[1, 1] == 2
    assert counts.loc[1, 2] == 1
    assert counts.loc[1, 0] == 1
    assert counts.loc[2, 2] == 3
    assert counts.loc[2, 1] == 1


def test_transition_probabilities_sum_to_one_by_observed_regime() -> None:
    summary = build_regime_transition_summary(make_regime_labels())

    row_sums = summary.transition_probabilities.sum(axis=1)

    assert all(abs(row_sum - 1.0) < 1e-12 for row_sum in row_sums)


def test_regime_duration_table_tracks_consecutive_blocks() -> None:
    summary = build_regime_transition_summary(make_regime_labels())

    durations = summary.regime_durations

    assert durations.iloc[0]["regime"] == 0
    assert durations.iloc[0]["duration"] == 3
    assert durations.iloc[2]["regime"] == 2
    assert durations.iloc[2]["duration"] == 4


def test_most_likely_next_regime_table() -> None:
    summary = build_regime_transition_summary(make_regime_labels())

    next_regimes = summary.most_likely_next_regime.set_index("current_regime")

    assert next_regimes.loc[0, "most_likely_next_regime"] == 0
    assert next_regimes.loc[1, "most_likely_next_regime"] == 1
    assert next_regimes.loc[2, "most_likely_next_regime"] == 2


def test_regime_transitions_rejects_empty_labels() -> None:
    with pytest.raises(RegimeTransitionError, match="Regime labels"):
        build_regime_transition_summary(pd.Series(dtype=int))


def test_regime_transitions_rejects_non_datetime_index() -> None:
    labels = pd.Series([0, 1, 0, 1], name="regime")

    with pytest.raises(RegimeTransitionError, match="DatetimeIndex"):
        build_regime_transition_summary(labels)


def test_regime_transitions_rejects_missing_labels() -> None:
    labels = make_regime_labels().astype(float)
    labels.iloc[0] = pd.NA

    with pytest.raises(RegimeTransitionError, match="missing"):
        build_regime_transition_summary(labels)


def test_regime_transitions_rejects_single_regime() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    labels = pd.Series([0, 0, 0, 0, 0], index=dates, name="regime")

    with pytest.raises(RegimeTransitionError, match="At least two unique"):
        build_regime_transition_summary(labels)


def test_regime_transitions_rejects_single_observation() -> None:
    dates = pd.date_range("2020-01-01", periods=1, freq="D")
    labels = pd.Series([0], index=dates, name="regime")

    with pytest.raises(RegimeTransitionError, match="At least two regime observations"):
        build_regime_transition_summary(labels)
