from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from regime_risk_engine.backtesting.comparison import StrategyComparisonResult
from regime_risk_engine.backtesting.diagnostics import BacktestDiagnostics
from regime_risk_engine.backtesting.regime_evaluation import RegimeBacktestEvaluation
from regime_risk_engine.reporting.export import ReportExportResult, export_report_bundle
from regime_risk_engine.reporting.plots import (
    ReportFigureBundle,
    build_report_figure_bundle,
    plot_strategy_rolling_metric,
)
from regime_risk_engine.reporting.tables import (
    ReportTableBundle,
    build_report_table_bundle,
)
from regime_risk_engine.validation.model_selection import RegimeModelSelectionReport


class ReportBuilderError(ValueError):
    """Raised when report assembly cannot be completed."""


@dataclass(frozen=True, slots=True)
class AssembledReport:
    """Container for assembled report tables and figures."""

    table_bundle: ReportTableBundle
    figure_bundle: ReportFigureBundle
    tables: dict[str, pd.DataFrame]
    figures: dict[str, Figure]


def assemble_core_report(
    strategy_comparison: StrategyComparisonResult,
    diagnostics: BacktestDiagnostics,
    regime_evaluation: RegimeBacktestEvaluation,
    model_selection: RegimeModelSelectionReport,
    ranking_metric: str | None = None,
    metric_labels: Mapping[str, str] | None = None,
) -> AssembledReport:
    """Assemble report-ready tables and figures from analytics outputs."""
    _validate_strategy_comparison(strategy_comparison)
    _validate_diagnostics(diagnostics)
    _validate_regime_evaluation(regime_evaluation)
    _validate_model_selection(model_selection)

    table_bundle = build_report_table_bundle(
        strategy_metric_summary=strategy_comparison.metric_summary,
        metric_deltas=strategy_comparison.metric_deltas,
        regime_metric_summary=regime_evaluation.metric_summary,
        model_ranking=model_selection.ranking,
        metric_labels=metric_labels,
    )

    figure_bundle = build_report_figure_bundle(
        cumulative_returns=diagnostics.cumulative_returns,
        drawdowns=diagnostics.drawdowns,
        metric_delta_table=table_bundle.metric_delta_table,
        model_ranking_table=table_bundle.model_ranking_table,
        ranking_metric=ranking_metric,
    )

    rolling_volatility_figure = plot_strategy_rolling_metric(
        rolling_metric=diagnostics.rolling_volatility,
        metric_label="Rolling Volatility",
    )
    rolling_sharpe_figure = plot_strategy_rolling_metric(
        rolling_metric=diagnostics.rolling_sharpe,
        metric_label="Rolling Sharpe Ratio",
    )

    tables = {
        "strategy_metric_table": table_bundle.strategy_metric_table,
        "metric_delta_table": table_bundle.metric_delta_table,
        "regime_metric_table": table_bundle.regime_metric_table,
        "model_ranking_table": table_bundle.model_ranking_table,
    }

    figures = {
        "cumulative_returns": figure_bundle.cumulative_returns,
        "drawdowns": figure_bundle.drawdowns,
        "metric_deltas": figure_bundle.metric_deltas,
        "model_ranking": figure_bundle.model_ranking,
        "rolling_volatility": rolling_volatility_figure,
        "rolling_sharpe": rolling_sharpe_figure,
    }

    return AssembledReport(
        table_bundle=table_bundle,
        figure_bundle=figure_bundle,
        tables=tables,
        figures=figures,
    )


def export_core_report(
    strategy_comparison: StrategyComparisonResult,
    diagnostics: BacktestDiagnostics,
    regime_evaluation: RegimeBacktestEvaluation,
    model_selection: RegimeModelSelectionReport,
    output_dir: Path | str,
    title: str = "Regime Risk Engine Report",
    ranking_metric: str | None = None,
    metric_labels: Mapping[str, str] | None = None,
    dpi: int = 150,
) -> ReportExportResult:
    """Assemble and export the core report artifacts."""
    assembled_report = assemble_core_report(
        strategy_comparison=strategy_comparison,
        diagnostics=diagnostics,
        regime_evaluation=regime_evaluation,
        model_selection=model_selection,
        ranking_metric=ranking_metric,
        metric_labels=metric_labels,
    )

    return export_report_bundle(
        tables=assembled_report.tables,
        figures=assembled_report.figures,
        output_dir=output_dir,
        title=title,
        dpi=dpi,
    )


def _validate_strategy_comparison(
    strategy_comparison: StrategyComparisonResult,
) -> None:
    _validate_frame(
        strategy_comparison.return_comparison,
        "strategy return comparison",
    )
    _validate_frame(
        strategy_comparison.metric_summary,
        "strategy metric summary",
    )
    _validate_frame(
        strategy_comparison.metric_deltas,
        "strategy metric deltas",
    )


def _validate_diagnostics(diagnostics: BacktestDiagnostics) -> None:
    _validate_frame(
        diagnostics.cumulative_returns,
        "diagnostic cumulative returns",
    )
    _validate_frame(
        diagnostics.drawdowns,
        "diagnostic drawdowns",
    )
    _validate_frame(
        diagnostics.rolling_volatility,
        "diagnostic rolling volatility",
    )
    _validate_frame(
        diagnostics.rolling_sharpe,
        "diagnostic rolling Sharpe",
    )


def _validate_regime_evaluation(
    regime_evaluation: RegimeBacktestEvaluation,
) -> None:
    _validate_frame(
        regime_evaluation.regime_return_frame,
        "regime return frame",
    )
    _validate_frame(
        regime_evaluation.metric_summary,
        "regime metric summary",
    )
    _validate_frame(
        regime_evaluation.metric_deltas,
        "regime metric deltas",
    )


def _validate_model_selection(
    model_selection: RegimeModelSelectionReport,
) -> None:
    _validate_frame(
        model_selection.model_summary,
        "model selection summary",
    )
    _validate_frame(
        model_selection.split_summary,
        "model selection split summary",
    )
    _validate_frame(
        model_selection.ranking,
        "model selection ranking",
    )


def _validate_frame(frame: pd.DataFrame, frame_name: str) -> None:
    if frame.empty:
        raise ReportBuilderError(f"{frame_name} cannot be empty")
