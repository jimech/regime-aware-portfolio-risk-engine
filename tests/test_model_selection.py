import pandas as pd
import pytest

from regime_risk_engine.validation.model_selection import (
    RegimeModelSelectionError,
    RegimeModelSelectionReport,
    compare_walk_forward_validations,
    rank_model_summaries,
)
from regime_risk_engine.validation.walk_forward import WalkForwardRegimeValidation


def make_label_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split_id": [0, 0, 0, 1, 1, 1],
            "sample": ["train", "train", "test", "train", "train", "test"],
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-01-06",
                ]
            ),
            "regime": [0, 1, 1, 0, 0, 1],
        }
    )


def make_split_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split_id": [0, 1],
            "train_start": pd.to_datetime(["2020-01-01", "2020-01-04"]),
            "train_end": pd.to_datetime(["2020-01-02", "2020-01-05"]),
            "test_start": pd.to_datetime(["2020-01-03", "2020-01-06"]),
            "test_end": pd.to_datetime(["2020-01-03", "2020-01-06"]),
            "train_observation_count": [2, 2],
            "test_observation_count": [1, 1],
            "train_regime_count": [2, 1],
            "test_regime_count": [1, 1],
            "train_transition_rate": [1.0, 0.0],
            "test_transition_rate": [float("nan"), float("nan")],
        }
    )


def make_diagnostic_summary(
    test_silhouette_score: float,
    test_transition_rate: float = 0.25,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split_id": [0, 0, 1, 1],
            "sample": ["train", "test", "train", "test"],
            "observation_count": [5, 3, 8, 3],
            "regime_count": [2, 2, 2, 2],
            "transition_rate": [0.50, test_transition_rate, 0.25, test_transition_rate],
            "dominant_regime_share": [0.60, 0.67, 0.75, 0.67],
            "silhouette_score": [
                0.40,
                test_silhouette_score,
                0.45,
                test_silhouette_score,
            ],
            "calinski_harabasz_score": [
                10.0,
                8.0,
                12.0,
                9.0,
            ],
        }
    )


def make_validation(
    test_silhouette_score: float,
    test_transition_rate: float = 0.25,
) -> WalkForwardRegimeValidation:
    return WalkForwardRegimeValidation(
        label_frame=make_label_frame(),
        split_summary=make_split_summary(),
        diagnostic_summary=make_diagnostic_summary(
            test_silhouette_score=test_silhouette_score,
            test_transition_rate=test_transition_rate,
        ),
    )


def test_compare_walk_forward_validations() -> None:
    report = compare_walk_forward_validations(
        {
            "kmeans": make_validation(test_silhouette_score=0.25),
            "gmm": make_validation(test_silhouette_score=0.40),
        }
    )

    assert isinstance(report, RegimeModelSelectionReport)
    assert set(report.model_summary["model"]) == {"kmeans", "gmm"}
    assert set(report.split_summary["model"]) == {"kmeans", "gmm"}
    assert report.ranking.iloc[0]["model"] == "gmm"
    assert report.ranking.iloc[1]["model"] == "kmeans"


def test_model_summary_contains_expected_columns() -> None:
    report = compare_walk_forward_validations(
        {
            "kmeans": make_validation(test_silhouette_score=0.25),
            "gmm": make_validation(test_silhouette_score=0.40),
        }
    )

    assert "split_count" in report.model_summary.columns
    assert "test_regime_count_mean" in report.model_summary.columns
    assert "test_transition_rate_mean" in report.model_summary.columns
    assert "test_dominant_regime_share_mean" in report.model_summary.columns
    assert "test_silhouette_score_mean" in report.model_summary.columns
    assert "test_calinski_harabasz_score_mean" in report.model_summary.columns


def test_compare_walk_forward_validations_ranks_lower_metric_better() -> None:
    report = compare_walk_forward_validations(
        {
            "low_turnover": make_validation(
                test_silhouette_score=0.25,
                test_transition_rate=0.10,
            ),
            "high_turnover": make_validation(
                test_silhouette_score=0.40,
                test_transition_rate=0.50,
            ),
        },
        ranking_metric="test_transition_rate_mean",
        higher_is_better=False,
    )

    assert report.ranking.iloc[0]["model"] == "low_turnover"
    assert report.ranking.iloc[1]["model"] == "high_turnover"


def test_rank_model_summaries() -> None:
    model_summary = pd.DataFrame(
        {
            "model": ["kmeans", "gmm"],
            "split_count": [2, 2],
            "test_silhouette_score_mean": [0.25, 0.40],
            "test_regime_count_mean": [2.0, 2.0],
            "test_transition_rate_mean": [0.25, 0.20],
            "test_dominant_regime_share_mean": [0.67, 0.60],
        }
    )

    ranking = rank_model_summaries(
        model_summary=model_summary,
        ranking_metric="test_silhouette_score_mean",
    )

    assert ranking["rank"].tolist() == [1, 2]
    assert ranking["model"].tolist() == ["gmm", "kmeans"]


def test_compare_walk_forward_validations_rejects_empty_mapping() -> None:
    with pytest.raises(RegimeModelSelectionError, match="At least one"):
        compare_walk_forward_validations({})


def test_compare_walk_forward_validations_rejects_empty_model_name() -> None:
    with pytest.raises(RegimeModelSelectionError, match="Model names"):
        compare_walk_forward_validations(
            {
                "": make_validation(test_silhouette_score=0.25),
            }
        )


def test_compare_walk_forward_validations_rejects_empty_label_frame() -> None:
    validation = WalkForwardRegimeValidation(
        label_frame=pd.DataFrame(),
        split_summary=make_split_summary(),
        diagnostic_summary=make_diagnostic_summary(test_silhouette_score=0.25),
    )

    with pytest.raises(RegimeModelSelectionError, match="label frame"):
        compare_walk_forward_validations({"kmeans": validation})


def test_compare_walk_forward_validations_rejects_missing_diagnostic_columns() -> None:
    validation = WalkForwardRegimeValidation(
        label_frame=make_label_frame(),
        split_summary=make_split_summary(),
        diagnostic_summary=make_diagnostic_summary(test_silhouette_score=0.25).drop(
            columns=["silhouette_score"]
        ),
    )

    with pytest.raises(RegimeModelSelectionError, match="Missing diagnostic"):
        compare_walk_forward_validations({"kmeans": validation})


def test_compare_walk_forward_validations_rejects_missing_split_columns() -> None:
    validation = WalkForwardRegimeValidation(
        label_frame=make_label_frame(),
        split_summary=make_split_summary().drop(columns=["test_end"]),
        diagnostic_summary=make_diagnostic_summary(test_silhouette_score=0.25),
    )

    with pytest.raises(RegimeModelSelectionError, match="Missing split"):
        compare_walk_forward_validations({"kmeans": validation})


def test_rank_model_summaries_rejects_missing_metric() -> None:
    model_summary = pd.DataFrame(
        {
            "model": ["kmeans"],
            "split_count": [2],
        }
    )

    with pytest.raises(RegimeModelSelectionError, match="Ranking metric"):
        rank_model_summaries(
            model_summary=model_summary,
            ranking_metric="test_silhouette_score_mean",
        )


def test_rank_model_summaries_rejects_all_missing_metric_values() -> None:
    model_summary = pd.DataFrame(
        {
            "model": ["kmeans", "gmm"],
            "split_count": [2, 2],
            "test_silhouette_score_mean": [float("nan"), float("nan")],
            "test_regime_count_mean": [2.0, 2.0],
            "test_transition_rate_mean": [0.25, 0.20],
            "test_dominant_regime_share_mean": [0.67, 0.60],
        }
    )

    with pytest.raises(RegimeModelSelectionError, match="only missing values"):
        rank_model_summaries(
            model_summary=model_summary,
            ranking_metric="test_silhouette_score_mean",
        )
