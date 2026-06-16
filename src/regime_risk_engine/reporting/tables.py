from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


class ReportingTableError(ValueError):
    """Raised when report-ready tables cannot be built."""


@dataclass(frozen=True, slots=True)
class ReportTableBundle:
    """Container for report-ready tables."""

    strategy_metric_table: pd.DataFrame
    metric_delta_table: pd.DataFrame
    regime_metric_table: pd.DataFrame
    model_ranking_table: pd.DataFrame


DEFAULT_METRIC_LABELS: dict[str, str] = {
    "cumulative_return": "Cumulative Return",
    "annualized_return": "Annualized Return",
    "annualized_volatility": "Annualized Volatility",
    "sharpe_ratio": "Sharpe Ratio",
    "sortino_ratio": "Sortino Ratio",
    "max_drawdown": "Maximum Drawdown",
    "var": "Value at Risk",
    "cvar": "Conditional Value at Risk",
    "test_silhouette_score_mean": "Mean Test Silhouette Score",
    "test_transition_rate_mean": "Mean Test Transition Rate",
    "test_dominant_regime_share_mean": "Mean Test Dominant Regime Share",
    "test_regime_count_mean": "Mean Test Regime Count",
}


STRATEGY_METRIC_TABLE_COLUMNS = [
    "strategy",
    "metric",
    "metric_label",
    "value",
]

METRIC_DELTA_TABLE_COLUMNS = [
    "metric",
    "metric_label",
    "benchmark_strategy",
    "candidate_strategy",
    "benchmark_value",
    "candidate_value",
    "absolute_delta",
    "relative_delta",
]

REGIME_METRIC_TABLE_COLUMNS = [
    "regime",
    "strategy",
    "observation_count",
    "metric",
    "metric_label",
    "value",
]


def build_strategy_metric_table(
    metric_summary: pd.DataFrame,
    metric_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Convert strategy metric summary into long report-ready format."""
    _validate_strategy_metric_summary(metric_summary)

    labels = _build_metric_label_map(metric_labels)
    metric_columns = [str(column) for column in metric_summary.columns]

    rows: list[dict[str, object]] = []

    for strategy_name, metric_row in metric_summary.iterrows():
        strategy = str(strategy_name)

        for metric in metric_columns:
            rows.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "metric_label": labels.get(metric, _humanize_metric_name(metric)),
                    "value": _to_float(
                        metric_row[metric],
                        context=f"{strategy}/{metric}",
                    ),
                }
            )

    return pd.DataFrame(rows, columns=STRATEGY_METRIC_TABLE_COLUMNS)


def build_metric_delta_table(
    metric_deltas: pd.DataFrame,
    metric_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Prepare candidate-minus-benchmark metric deltas for reporting."""
    _validate_metric_deltas(metric_deltas)

    labels = _build_metric_label_map(metric_labels)
    clean_deltas = metric_deltas.copy()
    clean_deltas["metric"] = clean_deltas["metric"].astype(str)
    clean_deltas["metric_label"] = clean_deltas["metric"].map(
        lambda metric: labels.get(metric, _humanize_metric_name(metric))
    )

    for column in [
        "benchmark_value",
        "candidate_value",
        "absolute_delta",
        "relative_delta",
    ]:
        clean_deltas[column] = pd.to_numeric(clean_deltas[column], errors="coerce")

    if (
        clean_deltas[
            [
                "benchmark_value",
                "candidate_value",
                "absolute_delta",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ReportingTableError("Metric delta table contains non-numeric values")

    return clean_deltas[METRIC_DELTA_TABLE_COLUMNS].reset_index(drop=True)


def build_regime_metric_table(
    regime_metric_summary: pd.DataFrame,
    metric_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Convert regime/strategy metric summary into long report-ready format."""
    _validate_regime_metric_summary(regime_metric_summary)

    labels = _build_metric_label_map(metric_labels)
    metric_columns = [
        str(column)
        for column in regime_metric_summary.columns
        if str(column) not in {"regime", "strategy", "observation_count"}
    ]

    rows: list[dict[str, object]] = []

    for _, row in regime_metric_summary.iterrows():
        regime = int(row["regime"])
        strategy = str(row["strategy"])
        observation_count = int(row["observation_count"])

        for metric in metric_columns:
            rows.append(
                {
                    "regime": regime,
                    "strategy": strategy,
                    "observation_count": observation_count,
                    "metric": metric,
                    "metric_label": labels.get(metric, _humanize_metric_name(metric)),
                    "value": _to_float(
                        row[metric],
                        context=f"regime {regime}/{strategy}/{metric}",
                    ),
                }
            )

    return pd.DataFrame(rows, columns=REGIME_METRIC_TABLE_COLUMNS)


def build_model_ranking_table(
    model_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare model ranking output for reporting."""
    _validate_model_ranking(model_ranking)

    ranking = model_ranking.copy()
    ranking["rank"] = pd.to_numeric(ranking["rank"], errors="coerce")

    if ranking["rank"].isna().any():
        raise ReportingTableError("Model ranking contains non-numeric ranks")

    ranking["rank"] = ranking["rank"].astype(int)
    ranking["model"] = ranking["model"].astype(str)

    return ranking.sort_values("rank").reset_index(drop=True)


def build_report_table_bundle(
    strategy_metric_summary: pd.DataFrame,
    metric_deltas: pd.DataFrame,
    regime_metric_summary: pd.DataFrame,
    model_ranking: pd.DataFrame,
    metric_labels: Mapping[str, str] | None = None,
) -> ReportTableBundle:
    """Build all report-ready tables used by reporting and dashboard layers."""
    return ReportTableBundle(
        strategy_metric_table=build_strategy_metric_table(
            metric_summary=strategy_metric_summary,
            metric_labels=metric_labels,
        ),
        metric_delta_table=build_metric_delta_table(
            metric_deltas=metric_deltas,
            metric_labels=metric_labels,
        ),
        regime_metric_table=build_regime_metric_table(
            regime_metric_summary=regime_metric_summary,
            metric_labels=metric_labels,
        ),
        model_ranking_table=build_model_ranking_table(model_ranking),
    )


def _validate_strategy_metric_summary(metric_summary: pd.DataFrame) -> None:
    if metric_summary.empty:
        raise ReportingTableError("Strategy metric summary cannot be empty")

    if metric_summary.index.has_duplicates:
        raise ReportingTableError(
            "Strategy metric summary contains duplicate strategies"
        )

    if metric_summary.columns.has_duplicates:
        raise ReportingTableError("Strategy metric summary contains duplicate metrics")

    if metric_summary.index.isna().any():
        raise ReportingTableError("Strategy metric summary contains missing strategies")


def _validate_metric_deltas(metric_deltas: pd.DataFrame) -> None:
    required_columns = set(METRIC_DELTA_TABLE_COLUMNS).difference({"metric_label"})
    missing_columns = required_columns.difference(metric_deltas.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ReportingTableError(f"Missing metric delta column(s): {missing}")

    if metric_deltas.empty:
        raise ReportingTableError("Metric delta table cannot be empty")

    if metric_deltas["metric"].isna().any():
        raise ReportingTableError("Metric delta table contains missing metric names")


def _validate_regime_metric_summary(regime_metric_summary: pd.DataFrame) -> None:
    required_columns = {"regime", "strategy", "observation_count"}
    missing_columns = required_columns.difference(regime_metric_summary.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ReportingTableError(f"Missing regime metric column(s): {missing}")

    if regime_metric_summary.empty:
        raise ReportingTableError("Regime metric summary cannot be empty")

    if regime_metric_summary[["regime", "strategy"]].duplicated().any():
        raise ReportingTableError(
            "Regime metric summary contains duplicate regime/strategy rows"
        )

    metric_columns = [
        str(column)
        for column in regime_metric_summary.columns
        if str(column) not in {"regime", "strategy", "observation_count"}
    ]

    if not metric_columns:
        raise ReportingTableError(
            "Regime metric summary must contain at least one metric column"
        )


def _validate_model_ranking(model_ranking: pd.DataFrame) -> None:
    required_columns = {"rank", "model"}
    missing_columns = required_columns.difference(model_ranking.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ReportingTableError(f"Missing model ranking column(s): {missing}")

    if model_ranking.empty:
        raise ReportingTableError("Model ranking table cannot be empty")

    if model_ranking["model"].duplicated().any():
        raise ReportingTableError("Model ranking table contains duplicate models")


def _build_metric_label_map(
    metric_labels: Mapping[str, str] | None,
) -> dict[str, str]:
    labels = DEFAULT_METRIC_LABELS.copy()

    if metric_labels is not None:
        labels.update(
            {
                str(metric).strip(): str(label).strip()
                for metric, label in metric_labels.items()
            }
        )

    return labels


def _humanize_metric_name(metric: str) -> str:
    return metric.replace("_", " ").title()


def _to_float(value: object, context: str) -> float:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric_value):
        raise ReportingTableError(f"Value is not numeric for {context}")

    return float(numeric_value)
