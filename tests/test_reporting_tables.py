import pandas as pd
import pytest

from regime_risk_engine.reporting.tables import (
    ReportingTableError,
    ReportTableBundle,
    build_metric_delta_table,
    build_model_ranking_table,
    build_regime_metric_table,
    build_report_table_bundle,
    build_strategy_metric_table,
)


def make_strategy_metric_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cumulative_return": [0.10, 0.15],
            "sharpe_ratio": [1.0, 1.3],
            "max_drawdown": [-0.20, -0.15],
        },
        index=["static", "dynamic"],
    )


def make_metric_deltas() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["cumulative_return", "sharpe_ratio", "max_drawdown"],
            "benchmark_strategy": ["static", "static", "static"],
            "candidate_strategy": ["dynamic", "dynamic", "dynamic"],
            "benchmark_value": [0.10, 1.0, -0.20],
            "candidate_value": [0.15, 1.3, -0.15],
            "absolute_delta": [0.05, 0.30, 0.05],
            "relative_delta": [0.50, 0.30, 0.25],
        }
    )


def make_regime_metric_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regime": [0, 0, 1, 1],
            "strategy": ["static", "dynamic", "static", "dynamic"],
            "observation_count": [10, 10, 8, 8],
            "cumulative_return": [0.05, 0.07, -0.02, 0.01],
            "sharpe_ratio": [0.8, 1.1, -0.2, 0.3],
        }
    )


def make_model_ranking() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rank": [2, 1],
            "model": ["kmeans", "gmm"],
            "test_silhouette_score_mean": [0.25, 0.40],
            "split_count": [2, 2],
            "test_regime_count_mean": [2.0, 2.0],
            "test_transition_rate_mean": [0.25, 0.20],
            "test_dominant_regime_share_mean": [0.67, 0.60],
        }
    )


def test_build_strategy_metric_table() -> None:
    table = build_strategy_metric_table(make_strategy_metric_summary())

    assert list(table.columns) == [
        "strategy",
        "metric",
        "metric_label",
        "value",
    ]
    assert len(table) == 6
    assert set(table["strategy"]) == {"static", "dynamic"}

    sharpe_row = table[
        (table["strategy"] == "dynamic") & (table["metric"] == "sharpe_ratio")
    ].iloc[0]

    assert sharpe_row["metric_label"] == "Sharpe Ratio"
    assert sharpe_row["value"] == pytest.approx(1.3)


def test_build_strategy_metric_table_uses_custom_metric_labels() -> None:
    table = build_strategy_metric_table(
        make_strategy_metric_summary(),
        metric_labels={"sharpe_ratio": "Risk-Adjusted Return"},
    )

    sharpe_labels = table[table["metric"] == "sharpe_ratio"]["metric_label"]

    assert set(sharpe_labels) == {"Risk-Adjusted Return"}


def test_build_metric_delta_table() -> None:
    table = build_metric_delta_table(make_metric_deltas())

    assert list(table.columns) == [
        "metric",
        "metric_label",
        "benchmark_strategy",
        "candidate_strategy",
        "benchmark_value",
        "candidate_value",
        "absolute_delta",
        "relative_delta",
    ]
    assert len(table) == 3

    sharpe_row = table[table["metric"] == "sharpe_ratio"].iloc[0]

    assert sharpe_row["metric_label"] == "Sharpe Ratio"
    assert sharpe_row["absolute_delta"] == pytest.approx(0.30)


def test_build_regime_metric_table() -> None:
    table = build_regime_metric_table(make_regime_metric_summary())

    assert list(table.columns) == [
        "regime",
        "strategy",
        "observation_count",
        "metric",
        "metric_label",
        "value",
    ]
    assert len(table) == 8
    assert set(table["regime"]) == {0, 1}
    assert set(table["strategy"]) == {"static", "dynamic"}

    regime_row = table[
        (table["regime"] == 1)
        & (table["strategy"] == "dynamic")
        & (table["metric"] == "cumulative_return")
    ].iloc[0]

    assert regime_row["observation_count"] == 8
    assert regime_row["value"] == pytest.approx(0.01)


def test_build_model_ranking_table_sorts_by_rank() -> None:
    table = build_model_ranking_table(make_model_ranking())

    assert table["rank"].tolist() == [1, 2]
    assert table["model"].tolist() == ["gmm", "kmeans"]


def test_build_report_table_bundle() -> None:
    bundle = build_report_table_bundle(
        strategy_metric_summary=make_strategy_metric_summary(),
        metric_deltas=make_metric_deltas(),
        regime_metric_summary=make_regime_metric_summary(),
        model_ranking=make_model_ranking(),
    )

    assert isinstance(bundle, ReportTableBundle)
    assert len(bundle.strategy_metric_table) == 6
    assert len(bundle.metric_delta_table) == 3
    assert len(bundle.regime_metric_table) == 8
    assert len(bundle.model_ranking_table) == 2


def test_strategy_metric_table_rejects_empty_summary() -> None:
    with pytest.raises(ReportingTableError, match="cannot be empty"):
        build_strategy_metric_table(pd.DataFrame())


def test_strategy_metric_table_rejects_duplicate_strategies() -> None:
    metric_summary = pd.DataFrame(
        {
            "sharpe_ratio": [1.0, 1.1],
        },
        index=["static", "static"],
    )

    with pytest.raises(ReportingTableError, match="duplicate strategies"):
        build_strategy_metric_table(metric_summary)


def test_strategy_metric_table_rejects_non_numeric_values() -> None:
    metric_summary = pd.DataFrame(
        {
            "sharpe_ratio": ["bad"],
        },
        index=["static"],
    )

    with pytest.raises(ReportingTableError, match="not numeric"):
        build_strategy_metric_table(metric_summary)


def test_metric_delta_table_rejects_missing_columns() -> None:
    metric_deltas = make_metric_deltas().drop(columns=["absolute_delta"])

    with pytest.raises(ReportingTableError, match="Missing metric delta"):
        build_metric_delta_table(metric_deltas)


def test_metric_delta_table_rejects_non_numeric_values() -> None:
    metric_deltas = make_metric_deltas()
    metric_deltas["candidate_value"] = metric_deltas["candidate_value"].astype(object)
    metric_deltas.loc[0, "candidate_value"] = "bad"

    with pytest.raises(ReportingTableError, match="non-numeric"):
        build_metric_delta_table(metric_deltas)


def test_regime_metric_table_rejects_missing_columns() -> None:
    regime_summary = make_regime_metric_summary().drop(columns=["regime"])

    with pytest.raises(ReportingTableError, match="Missing regime metric"):
        build_regime_metric_table(regime_summary)


def test_regime_metric_table_rejects_duplicate_regime_strategy_rows() -> None:
    regime_summary = pd.concat(
        [
            make_regime_metric_summary(),
            make_regime_metric_summary().iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(ReportingTableError, match="duplicate regime/strategy"):
        build_regime_metric_table(regime_summary)


def test_model_ranking_table_rejects_missing_columns() -> None:
    ranking = make_model_ranking().drop(columns=["rank"])

    with pytest.raises(ReportingTableError, match="Missing model ranking"):
        build_model_ranking_table(ranking)


def test_model_ranking_table_rejects_duplicate_models() -> None:
    ranking = pd.DataFrame(
        {
            "rank": [1, 2],
            "model": ["kmeans", "kmeans"],
        }
    )

    with pytest.raises(ReportingTableError, match="duplicate models"):
        build_model_ranking_table(ranking)
