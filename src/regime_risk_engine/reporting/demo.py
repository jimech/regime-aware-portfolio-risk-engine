from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class DemoReportInputError(ValueError):
    """Raised when demo report inputs cannot be created."""


@dataclass(frozen=True, slots=True)
class DemoReportInputs:
    """Container for generated demo report input paths."""

    output_dir: Path
    table_paths: dict[str, Path]
    figure_paths: dict[str, Path]


def create_demo_report_inputs(output_dir: Path | str) -> DemoReportInputs:
    """Create demo CSV tables and PNG figures for CLI report export."""
    clean_output_dir = _prepare_output_dir(output_dir)

    table_dir = clean_output_dir / "tables"
    figure_dir = clean_output_dir / "figures"

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    table_paths = {
        "strategy_metrics": table_dir / "strategy_metrics.csv",
        "metric_deltas": table_dir / "metric_deltas.csv",
    }
    figure_paths = {
        "cumulative_returns": figure_dir / "cumulative_returns.png",
        "drawdowns": figure_dir / "drawdowns.png",
    }

    build_demo_strategy_metric_table().to_csv(
        table_paths["strategy_metrics"],
        index=False,
    )
    build_demo_metric_delta_table().to_csv(
        table_paths["metric_deltas"],
        index=False,
    )

    _write_cumulative_returns_figure(figure_paths["cumulative_returns"])
    _write_drawdowns_figure(figure_paths["drawdowns"])

    return DemoReportInputs(
        output_dir=clean_output_dir,
        table_paths=table_paths,
        figure_paths=figure_paths,
    )


def build_demo_strategy_metric_table() -> pd.DataFrame:
    """Build a small demo strategy metric table."""
    return pd.DataFrame(
        {
            "strategy": [
                "static",
                "static",
                "static",
                "dynamic",
                "dynamic",
                "dynamic",
            ],
            "metric": [
                "cumulative_return",
                "sharpe_ratio",
                "max_drawdown",
                "cumulative_return",
                "sharpe_ratio",
                "max_drawdown",
            ],
            "metric_label": [
                "Cumulative Return",
                "Sharpe Ratio",
                "Maximum Drawdown",
                "Cumulative Return",
                "Sharpe Ratio",
                "Maximum Drawdown",
            ],
            "value": [
                0.08,
                0.90,
                -0.18,
                0.12,
                1.15,
                -0.14,
            ],
        }
    )


def build_demo_metric_delta_table() -> pd.DataFrame:
    """Build a small demo metric delta table."""
    return pd.DataFrame(
        {
            "metric": [
                "cumulative_return",
                "sharpe_ratio",
                "max_drawdown",
            ],
            "metric_label": [
                "Cumulative Return",
                "Sharpe Ratio",
                "Maximum Drawdown",
            ],
            "benchmark_strategy": [
                "static",
                "static",
                "static",
            ],
            "candidate_strategy": [
                "dynamic",
                "dynamic",
                "dynamic",
            ],
            "benchmark_value": [
                0.08,
                0.90,
                -0.18,
            ],
            "candidate_value": [
                0.12,
                1.15,
                -0.14,
            ],
            "absolute_delta": [
                0.04,
                0.25,
                0.04,
            ],
            "relative_delta": [
                0.50,
                0.2777777778,
                0.2222222222,
            ],
        }
    )


def _write_cumulative_returns_figure(path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="ME")
    cumulative_returns = pd.DataFrame(
        {
            "static": [
                0.00,
                0.02,
                0.01,
                0.04,
                0.06,
                0.08,
            ],
            "dynamic": [
                0.00,
                0.025,
                0.03,
                0.065,
                0.09,
                0.12,
            ],
        },
        index=dates,
    )

    figure, axis = plt.subplots(figsize=(8, 4))

    for column in cumulative_returns.columns:
        axis.plot(
            cumulative_returns.index,
            cumulative_returns[column],
            label=str(column),
        )

    axis.set_title("Demo Cumulative Returns")
    axis.set_xlabel("Date")
    axis.set_ylabel("Cumulative Return")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _write_drawdowns_figure(path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="ME")
    drawdowns = pd.DataFrame(
        {
            "static": [
                0.00,
                -0.02,
                -0.05,
                -0.03,
                -0.01,
                0.00,
            ],
            "dynamic": [
                0.00,
                -0.01,
                -0.025,
                -0.015,
                -0.005,
                0.00,
            ],
        },
        index=dates,
    )

    figure, axis = plt.subplots(figsize=(8, 4))

    for column in drawdowns.columns:
        axis.plot(
            drawdowns.index,
            drawdowns[column],
            label=str(column),
        )

    axis.axhline(0.0, linewidth=1.0)
    axis.set_title("Demo Drawdowns")
    axis.set_xlabel("Date")
    axis.set_ylabel("Drawdown")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def _prepare_output_dir(output_dir: Path | str) -> Path:
    clean_output_dir = Path(output_dir).expanduser().resolve()

    if clean_output_dir.exists() and not clean_output_dir.is_dir():
        raise DemoReportInputError("Output path exists and is not a directory")

    clean_output_dir.mkdir(parents=True, exist_ok=True)

    return clean_output_dir
