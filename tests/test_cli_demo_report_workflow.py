import json
from pathlib import Path

import pytest

from regime_risk_engine.cli import main
from regime_risk_engine.reporting.demo_workflow import (
    DemoReportWorkflowError,
    DemoReportWorkflowResult,
    run_demo_report_workflow,
)


def test_run_demo_report_workflow(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo-workflow"

    result = run_demo_report_workflow(output_dir=output_dir)

    assert isinstance(result, DemoReportWorkflowResult)
    assert result.output_dir == output_dir.resolve()
    assert result.input_dir.exists()
    assert result.report_dir.exists()
    assert result.markdown_path.exists()
    assert result.manifest_path.exists()

    assert set(result.table_paths) == {"strategy_metrics", "metric_deltas"}
    assert set(result.figure_paths) == {"cumulative_returns", "drawdowns"}

    assert (output_dir / "inputs" / "tables" / "strategy_metrics.csv").exists()
    assert (output_dir / "inputs" / "tables" / "metric_deltas.csv").exists()
    assert (output_dir / "inputs" / "figures" / "cumulative_returns.png").exists()
    assert (output_dir / "inputs" / "figures" / "drawdowns.png").exists()

    assert (output_dir / "report" / "tables" / "strategy_metrics.csv").exists()
    assert (output_dir / "report" / "tables" / "metric_deltas.csv").exists()
    assert (output_dir / "report" / "figures" / "cumulative_returns.png").exists()
    assert (output_dir / "report" / "figures" / "drawdowns.png").exists()


def test_demo_report_workflow_result_to_dict(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo-workflow"

    result = run_demo_report_workflow(output_dir=output_dir)
    result_dict = result.to_dict()

    assert result_dict["output_dir"] == str(output_dir.resolve())
    assert result_dict["input_dir"] == str((output_dir / "inputs").resolve())
    assert result_dict["report_dir"] == str((output_dir / "report").resolve())
    assert set(result_dict["tables"]) == {"strategy_metrics", "metric_deltas"}
    assert set(result_dict["figures"]) == {"cumulative_returns", "drawdowns"}


def test_cli_run_demo_report_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "demo-workflow"

    exit_code = main(
        [
            "run-demo-report",
            "--output-dir",
            str(output_dir),
            "--title",
            "Demo Report",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Demo report workflow completed" in captured.out
    assert (output_dir / "report" / "report_index.md").exists()
    assert (output_dir / "report" / "manifest.json").exists()


def test_cli_run_demo_report_json_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "demo-workflow"

    exit_code = main(
        [
            "run-demo-report",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert exit_code == 0
    assert result["output_dir"] == str(output_dir.resolve())
    assert result["input_dir"] == str((output_dir / "inputs").resolve())
    assert result["report_dir"] == str((output_dir / "report").resolve())
    assert "markdown_path" in result
    assert "manifest_path" in result


def test_run_demo_report_workflow_rejects_output_file(tmp_path: Path) -> None:
    output_file = tmp_path / "not_a_directory"
    output_file.write_text("content", encoding="utf-8")

    with pytest.raises(DemoReportWorkflowError, match="not a directory"):
        run_demo_report_workflow(output_dir=output_file)


def test_run_demo_report_workflow_rejects_empty_title(tmp_path: Path) -> None:
    with pytest.raises(DemoReportWorkflowError, match="title"):
        run_demo_report_workflow(
            output_dir=tmp_path / "demo-workflow",
            title=" ",
        )


def test_cli_run_demo_report_rejects_output_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_file = tmp_path / "not_a_directory"
    output_file.write_text("content", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main(
            [
                "run-demo-report",
                "--output-dir",
                str(output_file),
            ]
        )

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "not a directory" in captured.err
