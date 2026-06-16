import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from regime_risk_engine.cli import main
from regime_risk_engine.reporting.cli_export import (
    CliReportExportError,
    export_report_from_files,
)


def make_table_csv(tmp_path: Path) -> Path:
    path = tmp_path / "strategy_metrics.csv"
    pd.DataFrame(
        {
            "strategy": ["static", "dynamic"],
            "metric": ["sharpe_ratio", "sharpe_ratio"],
            "value": [1.0, 1.2],
        }
    ).to_csv(path, index=False)

    return path


def make_figure_png(tmp_path: Path) -> Path:
    path = tmp_path / "cumulative_returns.png"

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    figure.savefig(path)
    plt.close(figure)

    return path


def test_export_report_from_files_exports_tables_and_figures(tmp_path: Path) -> None:
    table_path = make_table_csv(tmp_path)
    figure_path = make_figure_png(tmp_path)
    output_dir = tmp_path / "report"

    result = export_report_from_files(
        table_specs=[f"strategy_metrics={table_path}"],
        figure_specs=[f"cumulative_returns={figure_path}"],
        output_dir=output_dir,
        title="CLI Report",
    )

    assert result.output_dir == output_dir.resolve()
    assert result.table_paths["strategy_metrics"].exists()
    assert result.figure_paths["cumulative_returns"].exists()
    assert result.markdown_path.exists()
    assert result.manifest_path.exists()


def test_cli_export_report_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    table_path = make_table_csv(tmp_path)
    figure_path = make_figure_png(tmp_path)
    output_dir = tmp_path / "report"

    exit_code = main(
        [
            "export-report",
            "--output-dir",
            str(output_dir),
            "--table",
            f"strategy_metrics={table_path}",
            "--figure",
            f"cumulative_returns={figure_path}",
            "--title",
            "CLI Report",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Report exported" in captured.out
    assert (output_dir / "tables" / "strategy_metrics.csv").exists()
    assert (output_dir / "figures" / "cumulative_returns.png").exists()
    assert (output_dir / "report_index.md").exists()
    assert (output_dir / "manifest.json").exists()


def test_cli_export_report_json_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    table_path = make_table_csv(tmp_path)
    output_dir = tmp_path / "report"

    exit_code = main(
        [
            "export-report",
            "--output-dir",
            str(output_dir),
            "--table",
            f"strategy_metrics={table_path}",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    result = json.loads(captured.out)

    assert exit_code == 0
    assert result["output_dir"] == str(output_dir.resolve())
    assert "strategy_metrics" in result["tables"]
    assert result["figures"] == {}


def test_export_report_from_files_rejects_missing_table_specs(tmp_path: Path) -> None:
    with pytest.raises(CliReportExportError, match="At least one --table"):
        export_report_from_files(
            table_specs=[],
            figure_specs=[],
            output_dir=tmp_path / "report",
        )


def test_export_report_from_files_rejects_bad_artifact_format(tmp_path: Path) -> None:
    with pytest.raises(CliReportExportError, match="name=path"):
        export_report_from_files(
            table_specs=["bad-format"],
            figure_specs=[],
            output_dir=tmp_path / "report",
        )


def test_export_report_from_files_rejects_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(CliReportExportError, match="does not exist"):
        export_report_from_files(
            table_specs=[f"strategy_metrics={missing_path}"],
            figure_specs=[],
            output_dir=tmp_path / "report",
        )


def test_export_report_from_files_rejects_non_csv_table(tmp_path: Path) -> None:
    table_path = tmp_path / "strategy_metrics.txt"
    table_path.write_text("not,csv", encoding="utf-8")

    with pytest.raises(CliReportExportError, match="CSV"):
        export_report_from_files(
            table_specs=[f"strategy_metrics={table_path}"],
            figure_specs=[],
            output_dir=tmp_path / "report",
        )


def test_export_report_from_files_rejects_non_png_figure(tmp_path: Path) -> None:
    table_path = make_table_csv(tmp_path)
    figure_path = tmp_path / "figure.jpg"
    figure_path.write_text("not a png", encoding="utf-8")

    with pytest.raises(CliReportExportError, match="PNG"):
        export_report_from_files(
            table_specs=[f"strategy_metrics={table_path}"],
            figure_specs=[f"figure={figure_path}"],
            output_dir=tmp_path / "report",
        )


def test_cli_export_report_rejects_missing_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "export-report",
                "--output-dir",
                str(tmp_path / "report"),
            ]
        )

    captured = capsys.readouterr()

    assert error.value.code == 2
    assert "At least one --table" in captured.err
