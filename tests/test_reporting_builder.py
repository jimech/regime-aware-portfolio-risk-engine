from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from regime_risk_engine.backtesting.comparison import StrategyComparisonResult
from regime_risk_engine.backtesting.diagnostics import BacktestDiagnostics
from regime_risk_engine.backtesting.engine import BacktestResult
from regime_risk_engine.backtesting.regime_evaluation import RegimeBacktestEvaluation
from regime_risk_engine.reporting.builder import (
    AssembledReport,
    assemble_core_report,
    export_core_report,
)
from regime_risk_engine.reporting.export import ReportExportResult
from regime_risk_engine.validation.model_selection import RegimeModelSelectionReport


def make_dates() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=4, freq="D")


def make_backtest_result(name: str) -> BacktestResult:
    dates = make_dates()
    returns = pd.Series(
        [0.01, 0.02, -0.01, 0.03],
        index=dates,
        name=name,
    )
    turnover = pd.Series(
        [0.0, 0.1, 0.2, 0.1],
        index=dates,
        name="turnover",
    )
    costs = pd.Series(
        [0.0, 0.0001, 0.0002, 0.0001],
        index=dates,
        name="transaction_cost",
    )
    weights = pd.DataFrame(
        {
            "SPY": [0.6, 0.6, 0.5, 0.5],
            "TLT": [0.4, 0.4, 0.5, 0.5],
        },
        index=dates,
    )

    return BacktestResult(
        gross_returns=returns,
        net_returns=returns - costs,
        turnover=turnover,
        transaction_costs=costs,
        applied_weights=weights,
    )


def make_strategy_comparison() -> StrategyComparisonResult:
    dates = make_dates()

    return_comparison = pd.DataFrame(
        {
            "static": [0.01, 0.02, -0.01, 0.03],
            "dynamic": [0.015, 0.018, 0.00, 0.035],
        },
        index=dates,
    )

    metric_summary = pd.DataFrame(
        {
            "cumulative_return": [0.05, 0.07],
            "sharpe_ratio": [1.0, 1.2],
            "max_drawdown": [-0.10, -0.08],
        },
        index=["static", "dynamic"],
    )

    metric_deltas = pd.DataFrame(
        {
            "metric": ["cumulative_return", "sharpe_ratio", "max_drawdown"],
            "benchmark_strategy": ["static", "static", "static"],
            "candidate_strategy": ["dynamic", "dynamic", "dynamic"],
            "benchmark_value": [0.05, 1.0, -0.10],
            "candidate_value": [0.07, 1.2, -0.08],
            "absolute_delta": [0.02, 0.2, 0.02],
            "relative_delta": [0.40, 0.20, 0.20],
        }
    )

    return StrategyComparisonResult(
        static_backtest=make_backtest_result("static"),
        dynamic_backtest=make_backtest_result("dynamic"),
        return_comparison=return_comparison,
        metric_summary=metric_summary,
        metric_deltas=metric_deltas,
    )


def make_diagnostics() -> BacktestDiagnostics:
    dates = make_dates()

    cumulative_returns = pd.DataFrame(
        {
            "static": [0.01, 0.03, 0.02, 0.05],
            "dynamic": [0.015, 0.033, 0.033, 0.07],
        },
        index=dates,
    )
    drawdowns = pd.DataFrame(
        {
            "static": [0.0, 0.0, -0.01, 0.0],
            "dynamic": [0.0, 0.0, 0.0, 0.0],
        },
        index=dates,
    )
    rolling_volatility = pd.DataFrame(
        {
            "static": [None, None, 0.15, 0.14],
            "dynamic": [None, None, 0.12, 0.11],
        },
        index=dates,
    )
    rolling_sharpe = pd.DataFrame(
        {
            "static": [None, None, 0.8, 0.9],
            "dynamic": [None, None, 1.1, 1.2],
        },
        index=dates,
    )

    return BacktestDiagnostics(
        cumulative_returns=cumulative_returns,
        drawdowns=drawdowns,
        rolling_volatility=rolling_volatility,
        rolling_sharpe=rolling_sharpe,
    )


def make_regime_evaluation() -> RegimeBacktestEvaluation:
    dates = make_dates()

    regime_return_frame = pd.DataFrame(
        {
            "static": [0.01, 0.02, -0.01, 0.03],
            "dynamic": [0.015, 0.018, 0.00, 0.035],
            "regime": [0, 0, 1, 1],
        },
        index=dates,
    )

    metric_summary = pd.DataFrame(
        {
            "regime": [0, 0, 1, 1],
            "strategy": ["static", "dynamic", "static", "dynamic"],
            "observation_count": [2, 2, 2, 2],
            "cumulative_return": [0.03, 0.034, 0.02, 0.035],
            "sharpe_ratio": [0.8, 1.0, 0.4, 0.7],
        }
    )

    metric_deltas = pd.DataFrame(
        {
            "regime": [0, 1],
            "metric": ["sharpe_ratio", "sharpe_ratio"],
            "benchmark_strategy": ["static", "static"],
            "candidate_strategy": ["dynamic", "dynamic"],
            "benchmark_value": [0.8, 0.4],
            "candidate_value": [1.0, 0.7],
            "absolute_delta": [0.2, 0.3],
            "relative_delta": [0.25, 0.75],
        }
    )

    return RegimeBacktestEvaluation(
        regime_return_frame=regime_return_frame,
        metric_summary=metric_summary,
        metric_deltas=metric_deltas,
    )


def make_model_selection() -> RegimeModelSelectionReport:
    model_summary = pd.DataFrame(
        {
            "model": ["kmeans", "gmm"],
            "split_count": [2, 2],
            "test_silhouette_score_mean": [0.25, 0.40],
        }
    )

    split_summary = pd.DataFrame(
        {
            "model": ["kmeans", "gmm"],
            "split_id": [0, 0],
            "test_regime_count": [2, 2],
        }
    )

    ranking = pd.DataFrame(
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

    return RegimeModelSelectionReport(
        model_summary=model_summary,
        split_summary=split_summary,
        ranking=ranking,
    )


def close_figures(figures: dict[str, Figure]) -> None:
    for figure in figures.values():
        plt.close(figure)


def test_assemble_core_report() -> None:
    report = assemble_core_report(
        strategy_comparison=make_strategy_comparison(),
        diagnostics=make_diagnostics(),
        regime_evaluation=make_regime_evaluation(),
        model_selection=make_model_selection(),
        ranking_metric="test_silhouette_score_mean",
    )

    assert isinstance(report, AssembledReport)
    assert set(report.tables) == {
        "strategy_metric_table",
        "metric_delta_table",
        "regime_metric_table",
        "model_ranking_table",
    }
    assert set(report.figures) == {
        "cumulative_returns",
        "drawdowns",
        "metric_deltas",
        "model_ranking",
        "rolling_volatility",
        "rolling_sharpe",
    }
    assert len(report.tables["strategy_metric_table"]) == 6
    assert len(report.tables["metric_delta_table"]) == 3
    assert isinstance(report.figures["cumulative_returns"], Figure)

    close_figures(report.figures)


def test_export_core_report(tmp_path: Path) -> None:
    result = export_core_report(
        strategy_comparison=make_strategy_comparison(),
        diagnostics=make_diagnostics(),
        regime_evaluation=make_regime_evaluation(),
        model_selection=make_model_selection(),
        output_dir=tmp_path,
        title="Core Report",
        ranking_metric="test_silhouette_score_mean",
        dpi=80,
    )

    assert isinstance(result, ReportExportResult)
    assert result.markdown_path.exists()
    assert result.manifest_path.exists()
    assert set(result.table_paths) == {
        "strategy_metric_table",
        "metric_delta_table",
        "regime_metric_table",
        "model_ranking_table",
    }
    assert set(result.figure_paths) == {
        "cumulative_returns",
        "drawdowns",
        "metric_deltas",
        "model_ranking",
        "rolling_volatility",
        "rolling_sharpe",
    }

    for path in result.table_paths.values():
        assert path.exists()

    for path in result.figure_paths.values():
        assert path.exists()
