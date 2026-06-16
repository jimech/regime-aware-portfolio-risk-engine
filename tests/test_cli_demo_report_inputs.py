import json
from pathlib import Path

import pandas as pd
import pytest

from regime_risk_engine.cli import main
from regime_risk_engine.reporting.demo import (
    DemoReportInputError,
    build_demo_metric_delta_table,
    build_demo_strategy_metric_table,
    create_demo_report_inputs,
)


def test_build_demo_strategy_metric_table() -> None:
    table = build_demo_strategy_metric_table()

    assert list(table.columns) == [
        "strategy",
        "metric",
        "metric_label",
        "value",
    ]
    assert set(table["strategy"]) == {"static", "dynamic"}
    assert not table.empty


def test_build_demo_metric_delta_table() -> None:
    table = build_demo_metric_delta_table()

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
    assert not table.empty


def test_create_demo_report_inputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo-inputs"

    result = create_demo_report_inputs(output_dir=output_dir)

    assert result.output_dir == output_dir.resolve()
    assert set(result.table_paths) == {"strategy_metrics", "metric_deltas"}
    assert set(result.figure_paths) == {"cumulative_returns", "drawdowns"}

    for path in result.table_paths.values():
        assert path.exists()
        assert path.suffix == ".csv"

    for path in result.figure_paths.values():
        assert path.exists()
        assert path.suffix == ".png"

    strategy_metrics = pd.read_csv(result.table_paths["strategy_metrics"])

    assert not strategy_metrics.empty


def test_cli_create_demo_report_inputs_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "demo-inputs"

    exit_code = main(
        [
            "create-demo-report-inputs",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Demo report inputs created" in captured.out
    assert (output_dir / "tables" / "strategy_metrics.csv").exists()
    assert (output_dir / "tables" / "metric_deltas.csv").exists()
    assert (output_dir / "figures" / "cumulative_returns.png").exists()
    assert (output_dir / "figures" / "drawdowns.png").exists()


def test_cli_create_demo_report_inputs_json_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "demo-inputs"

    exit_code = main(
        [
            "create-demo-report-inputs",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert exit_code == 0
    assert result["output_dir"] == str(output_dir.resolve())
    assert set(result["tables"]) == {"strategy_metrics", "metric_deltas"}
    assert set(result["figures"]) == {"cumulative_returns", "drawdowns"}


def test_create_demo_report_inputs_rejects_output_file(tmp_path: Path) -> None:
    output_file = tmp_path / "not_a_directory"
    output_file.write_text("content", encoding="utf-8")

    with pytest.raises(DemoReportInputError, match="not a directory"):
        create_demo_report_inputs(output_dir=output_file)


def test_demo_inputs_can_feed_export_report_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "demo-inputs"
    output_dir = tmp_path / "demo-report"

    demo_inputs = create_demo_report_inputs(output_dir=input_dir)

    exit_code = main(
        [
            "export-report",
            "--output-dir",
            str(output_dir),
            "--table",
            f"strategy_metrics={demo_inputs.table_paths['strategy_metrics']}",
            "--table",
            f"metric_deltas={demo_inputs.table_paths['metric_deltas']}",
            "--figure",
            f"cumulative_returns={demo_inputs.figure_paths['cumulative_returns']}",
            "--figure",
            f"drawdowns={demo_inputs.figure_paths['drawdowns']}",
            "--title",
            "Demo Report",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Report exported" in captured.out
    assert (output_dir / "tables" / "strategy_metrics.csv").exists()
    assert (output_dir / "tables" / "metric_deltas.csv").exists()
    assert (output_dir / "figures" / "cumulative_returns.png").exists()
    assert (output_dir / "figures" / "drawdowns.png").exists()
    assert (output_dir / "report_index.md").exists()
    assert (output_dir / "manifest.json").exists()
