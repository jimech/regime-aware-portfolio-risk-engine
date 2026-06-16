import pandas as pd
import pytest

from regime_risk_engine.validation.regime_stability import (
    RegimeStabilityError,
    RegimeStabilityReport,
    build_regime_stability_report,
    calculate_label_agreement,
    calculate_regime_distribution,
    calculate_regime_transition_matrix,
    calculate_regime_transition_summary,
)


def make_labels() -> pd.Series:
    return pd.Series(
        [0, 0, 1, 1, 0, 0],
        index=pd.date_range("2020-01-01", periods=6, freq="D"),
        name="regime",
    )


def make_permuted_labels() -> pd.Series:
    return pd.Series(
        [1, 1, 0, 0, 1, 1],
        index=pd.date_range("2020-01-01", periods=6, freq="D"),
        name="regime",
    )


def make_candidate_labels() -> pd.Series:
    return pd.Series(
        [0, 1, 1, 1, 0, 0],
        index=pd.date_range("2020-01-01", periods=6, freq="D"),
        name="regime",
    )


def test_calculate_regime_distribution() -> None:
    distribution = calculate_regime_distribution(make_labels())

    assert distribution["regime"].tolist() == [0, 1]
    assert distribution["observation_count"].tolist() == [4, 2]
    assert distribution.loc[0, "observation_share"] == pytest.approx(4 / 6)
    assert distribution.loc[1, "observation_share"] == pytest.approx(2 / 6)


def test_calculate_regime_transition_matrix_counts() -> None:
    matrix = calculate_regime_transition_matrix(
        make_labels(),
        normalize=False,
    )

    assert matrix.loc[0, 0] == pytest.approx(2.0)
    assert matrix.loc[0, 1] == pytest.approx(1.0)
    assert matrix.loc[1, 0] == pytest.approx(1.0)
    assert matrix.loc[1, 1] == pytest.approx(1.0)


def test_calculate_regime_transition_matrix_normalized() -> None:
    matrix = calculate_regime_transition_matrix(
        make_labels(),
        normalize=True,
    )

    assert matrix.loc[0, 0] == pytest.approx(2 / 3)
    assert matrix.loc[0, 1] == pytest.approx(1 / 3)
    assert matrix.loc[1, 0] == pytest.approx(1 / 2)
    assert matrix.loc[1, 1] == pytest.approx(1 / 2)


def test_calculate_regime_transition_summary() -> None:
    summary = calculate_regime_transition_summary(make_labels())

    assert summary["observation_count"] == pytest.approx(6)
    assert summary["regime_count"] == pytest.approx(2)
    assert summary["transition_count"] == pytest.approx(2)
    assert summary["transition_rate"] == pytest.approx(2 / 5)
    assert summary["dominant_regime_share"] == pytest.approx(4 / 6)


def test_calculate_label_agreement_for_identical_labels() -> None:
    agreement = calculate_label_agreement(
        reference_labels=make_labels(),
        candidate_labels=make_labels(),
    )

    assert agreement["observation_count"] == pytest.approx(6)
    assert agreement["direct_match_rate"] == pytest.approx(1.0)
    assert agreement["adjusted_rand_index"] == pytest.approx(1.0)
    assert agreement["normalized_mutual_information"] == pytest.approx(1.0)


def test_calculate_label_agreement_is_label_invariant() -> None:
    agreement = calculate_label_agreement(
        reference_labels=make_labels(),
        candidate_labels=make_permuted_labels(),
    )

    assert agreement["direct_match_rate"] == pytest.approx(0.0)
    assert agreement["adjusted_rand_index"] == pytest.approx(1.0)
    assert agreement["normalized_mutual_information"] == pytest.approx(1.0)


def test_build_regime_stability_report() -> None:
    report = build_regime_stability_report(
        {
            "kmeans": make_labels(),
            "gmm": make_candidate_labels(),
        }
    )

    assert isinstance(report, RegimeStabilityReport)
    assert set(report.distribution_summary["model"]) == {"kmeans", "gmm"}
    assert set(report.transition_summary["model"]) == {"kmeans", "gmm"}
    assert len(report.agreement_summary) == 1
    assert report.agreement_summary.loc[0, "reference_model"] == "kmeans"
    assert report.agreement_summary.loc[0, "candidate_model"] == "gmm"


def test_build_regime_stability_report_with_reference_name() -> None:
    report = build_regime_stability_report(
        {
            "baseline": make_labels(),
            "candidate_a": make_candidate_labels(),
            "candidate_b": make_permuted_labels(),
        },
        reference_name="baseline",
    )

    assert len(report.agreement_summary) == 2
    assert set(report.agreement_summary["reference_model"]) == {"baseline"}
    assert set(report.agreement_summary["candidate_model"]) == {
        "candidate_a",
        "candidate_b",
    }


def test_build_regime_stability_report_allows_single_label_set() -> None:
    report = build_regime_stability_report(
        {
            "kmeans": make_labels(),
        }
    )

    assert len(report.distribution_summary) == 2
    assert len(report.transition_summary) == 1
    assert report.agreement_summary.empty


def test_regime_stability_rejects_empty_label_set_mapping() -> None:
    with pytest.raises(RegimeStabilityError, match="At least one"):
        build_regime_stability_report({})


def test_regime_stability_rejects_non_datetime_index() -> None:
    labels = pd.Series([0, 1], index=[0, 1], name="regime")

    with pytest.raises(RegimeStabilityError, match="DatetimeIndex"):
        calculate_regime_distribution(labels)


def test_regime_stability_rejects_duplicate_dates() -> None:
    labels = pd.Series(
        [0, 1],
        index=pd.to_datetime(["2020-01-01", "2020-01-01"]),
        name="regime",
    )

    with pytest.raises(RegimeStabilityError, match="duplicate"):
        calculate_regime_distribution(labels)


def test_regime_stability_rejects_missing_labels() -> None:
    labels = pd.Series(
        [0, None],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
        name="regime",
    )

    with pytest.raises(RegimeStabilityError, match="missing"):
        calculate_regime_distribution(labels)


def test_regime_stability_rejects_non_integer_labels() -> None:
    labels = pd.Series(
        [0.0, 1.5],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
        name="regime",
    )

    with pytest.raises(RegimeStabilityError, match="integer-like"):
        calculate_regime_distribution(labels)


def test_transition_matrix_rejects_single_observation() -> None:
    labels = pd.Series(
        [0],
        index=pd.to_datetime(["2020-01-01"]),
        name="regime",
    )

    with pytest.raises(RegimeStabilityError, match="At least two"):
        calculate_regime_transition_matrix(labels)


def test_label_agreement_rejects_no_overlapping_dates() -> None:
    reference = pd.Series(
        [0, 1],
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        name="regime",
    )
    candidate = pd.Series(
        [0, 1],
        index=pd.to_datetime(["2030-01-01", "2030-01-02"]),
        name="regime",
    )

    with pytest.raises(RegimeStabilityError, match="no overlapping dates"):
        calculate_label_agreement(reference, candidate)


def test_build_report_rejects_missing_reference_name() -> None:
    with pytest.raises(RegimeStabilityError, match="Reference model"):
        build_regime_stability_report(
            {
                "kmeans": make_labels(),
                "gmm": make_candidate_labels(),
            },
            reference_name="missing_model",
        )
