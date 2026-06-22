from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


class ReportingPlotError(ValueError):
    """Raised when report plots cannot be built."""


@dataclass(frozen=True, slots=True)
class ReportFigureBundle:
    """Container for report-ready figures."""

    cumulative_returns: Figure
    drawdowns: Figure
    metric_deltas: Figure
    model_ranking: Figure


def plot_strategy_cumulative_returns(
    cumulative_returns: pd.DataFrame,
    title: str = "Strategy Cumulative Returns",
) -> Figure:
    """Plot cumulative returns for one or more strategies."""
    frame = _validate_strategy_time_series_frame(
        cumulative_returns,
        frame_name="cumulative returns",
        allow_missing=False,
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    for column in frame.columns:
        ax.plot(frame.index, frame[column], label=str(column))

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_strategy_drawdowns(
    drawdowns: pd.DataFrame,
    title: str = "Strategy Drawdowns",
) -> Figure:
    """Plot drawdowns for one or more strategies."""
    frame = _validate_strategy_time_series_frame(
        drawdowns,
        frame_name="drawdowns",
        allow_missing=False,
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    for column in frame.columns:
        ax.plot(frame.index, frame[column], label=str(column))

    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_strategy_rolling_metric(
    rolling_metric: pd.DataFrame,
    metric_label: str,
    title: str | None = None,
) -> Figure:
    """Plot a rolling diagnostic metric for one or more strategies."""
    clean_metric_label = str(metric_label).strip()

    if not clean_metric_label:
        raise ReportingPlotError("Metric label must be non-empty")

    frame = _validate_strategy_time_series_frame(
        rolling_metric,
        frame_name=clean_metric_label,
        allow_missing=True,
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    for column in frame.columns:
        ax.plot(frame.index, frame[column], label=str(column))

    ax.set_title(title or clean_metric_label)
    ax.set_xlabel("Date")
    ax.set_ylabel(clean_metric_label)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_metric_delta_bars(
    metric_delta_table: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    value_col: str = "absolute_delta",
    title: str = "Dynamic vs Static Metric Deltas",
) -> Figure:
    """Plot metric deltas from a report-ready metric delta table."""
    clean_table = _validate_metric_delta_table(
        metric_delta_table=metric_delta_table,
        value_col=value_col,
    )
    selected_table = _filter_metric_delta_table(
        metric_delta_table=clean_table,
        metrics=metrics,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(selected_table["metric_label"], selected_table[value_col])
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    return fig


def plot_model_ranking(
    model_ranking_table: pd.DataFrame,
    ranking_metric: str | None = None,
    title: str = "Regime Model Ranking",
) -> Figure:
    """Plot model ranking by one selected ranking metric."""
    clean_table = _validate_model_ranking_table(model_ranking_table)
    clean_ranking_metric = _resolve_ranking_metric(
        model_ranking_table=clean_table,
        ranking_metric=ranking_metric,
    )

    clean_table[clean_ranking_metric] = pd.to_numeric(
        clean_table[clean_ranking_metric],
        errors="coerce",
    )

    if clean_table[clean_ranking_metric].isna().all():
        raise ReportingPlotError(
            f"Ranking metric contains only missing values: {clean_ranking_metric}"
        )

    clean_table = clean_table.sort_values("rank")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(clean_table["model"], clean_table[clean_ranking_metric])
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(clean_ranking_metric.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    return fig


def build_report_figure_bundle(
    cumulative_returns: pd.DataFrame,
    drawdowns: pd.DataFrame,
    metric_delta_table: pd.DataFrame,
    model_ranking_table: pd.DataFrame,
    ranking_metric: str | None = None,
) -> ReportFigureBundle:
    """Build the core figure bundle for reports and dashboards."""
    return ReportFigureBundle(
        cumulative_returns=plot_strategy_cumulative_returns(cumulative_returns),
        drawdowns=plot_strategy_drawdowns(drawdowns),
        metric_deltas=plot_metric_delta_bars(metric_delta_table),
        model_ranking=plot_model_ranking(
            model_ranking_table=model_ranking_table,
            ranking_metric=ranking_metric,
        ),
    )


def _validate_strategy_time_series_frame(
    frame: pd.DataFrame,
    frame_name: str,
    allow_missing: bool,
) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ReportingPlotError(f"{frame_name} index must be a DatetimeIndex")

    if frame.empty:
        raise ReportingPlotError(f"{frame_name} frame cannot be empty")

    if frame.index.has_duplicates:
        raise ReportingPlotError(f"{frame_name} frame contains duplicate dates")

    if frame.columns.has_duplicates:
        raise ReportingPlotError(f"{frame_name} frame contains duplicate columns")

    clean_frame = frame.copy().sort_index()

    for column in clean_frame.columns:
        clean_frame[column] = pd.to_numeric(clean_frame[column], errors="coerce")

    if allow_missing:
        if clean_frame.dropna(how="all").empty:
            raise ReportingPlotError(f"{frame_name} frame contains only missing values")
    elif clean_frame.isna().any().any():
        raise ReportingPlotError(f"{frame_name} frame contains missing values")

    return clean_frame


def _validate_metric_delta_table(
    metric_delta_table: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    allowed_value_columns = {"absolute_delta", "relative_delta"}

    if value_col not in allowed_value_columns:
        allowed = ", ".join(sorted(allowed_value_columns))
        raise ReportingPlotError(f"Value column must be one of: {allowed}")

    required_columns = {"metric", "metric_label", value_col}
    missing_columns = required_columns.difference(metric_delta_table.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ReportingPlotError(f"Missing metric delta column(s): {missing}")

    if metric_delta_table.empty:
        raise ReportingPlotError("Metric delta table cannot be empty")

    clean_table = metric_delta_table.copy()
    clean_table["metric"] = clean_table["metric"].astype(str)
    clean_table["metric_label"] = clean_table["metric_label"].astype(str)
    clean_table[value_col] = pd.to_numeric(clean_table[value_col], errors="coerce")

    if clean_table["metric"].str.strip().eq("").any():
        raise ReportingPlotError("Metric delta table contains empty metric names")

    if clean_table["metric_label"].str.strip().eq("").any():
        raise ReportingPlotError("Metric delta table contains empty metric labels")

    if clean_table[value_col].isna().any():
        raise ReportingPlotError("Metric delta table contains non-numeric values")

    return clean_table


def _filter_metric_delta_table(
    metric_delta_table: pd.DataFrame,
    metrics: Sequence[str] | None,
) -> pd.DataFrame:
    if metrics is None:
        return metric_delta_table.reset_index(drop=True)

    selected_metrics = {str(metric).strip() for metric in metrics}

    if not selected_metrics or "" in selected_metrics:
        raise ReportingPlotError("Selected metrics must be non-empty")

    selected_table = metric_delta_table[
        metric_delta_table["metric"].isin(selected_metrics)
    ].copy()

    if selected_table.empty:
        raise ReportingPlotError("No metric delta rows matched the selected metrics")

    return selected_table.reset_index(drop=True)


def _validate_model_ranking_table(model_ranking_table: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"rank", "model"}
    missing_columns = required_columns.difference(model_ranking_table.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ReportingPlotError(f"Missing model ranking column(s): {missing}")

    if model_ranking_table.empty:
        raise ReportingPlotError("Model ranking table cannot be empty")

    if model_ranking_table["model"].duplicated().any():
        raise ReportingPlotError("Model ranking table contains duplicate models")

    clean_table = model_ranking_table.copy()
    clean_table["rank"] = pd.to_numeric(clean_table["rank"], errors="coerce")

    if clean_table["rank"].isna().any():
        raise ReportingPlotError("Model ranking table contains non-numeric ranks")

    clean_table["rank"] = clean_table["rank"].astype(int)
    clean_table["model"] = clean_table["model"].astype(str)

    if clean_table["model"].str.strip().eq("").any():
        raise ReportingPlotError("Model ranking table contains empty model names")

    return clean_table


def _resolve_ranking_metric(
    model_ranking_table: pd.DataFrame,
    ranking_metric: str | None,
) -> str:
    if ranking_metric is not None:
        clean_metric = str(ranking_metric).strip()

        if not clean_metric:
            raise ReportingPlotError("Ranking metric must be non-empty")

        if clean_metric not in model_ranking_table.columns:
            raise ReportingPlotError(f"Ranking metric not found: {clean_metric}")

        return clean_metric

    excluded_columns = {"rank", "model"}

    for column in model_ranking_table.columns:
        clean_column = str(column)

        if clean_column in excluded_columns:
            continue

        values = pd.to_numeric(model_ranking_table[clean_column], errors="coerce")

        if not values.isna().all():
            return clean_column

    raise ReportingPlotError("No numeric ranking metric found")
