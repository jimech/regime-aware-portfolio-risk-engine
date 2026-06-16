import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from regime_risk_engine.reporting.export import (
    ReportExportError,
    ReportExportResult,
    export_report_bundle,
    export_report_figures,
    export_report_tables,
    write_report_manifest,
    write_report_markdown_index,
)


def make_tables() -> dict[str, pd.DataFrame]:
    return {
        "strategy_metrics": pd.DataFrame(
            {
                "strategy": ["static", "dynamic"],
                "metric": ["sharpe_ratio", "sharpe_ratio"],
                "value": [1.0, 1.2],
            }
        ),
        "metric_deltas": pd.DataFrame(
            {
                "metric": ["sharpe_ratio"],
                "absolute_delta": [0.2],
            }
        ),
    }


def make_figure() -> Figure:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    axis.set_title("Test Figure")

    return figure


def make_figures() -> dict[str, Figure]:
    return {
        "cumulative_returns": make_figure(),
        "drawdowns": make_figure(),
    }


def test_export_report_tables(tmp_path: Path) -> None:
    table_paths = export_report_tables(
        tables=make_tables(),
        output_dir=tmp_path,
    )

    assert set(table_paths) == {"strategy_metrics", "metric_deltas"}

    for path in table_paths.values():
        assert path.exists()
        assert path.suffix == ".csv"


def test_export_report_figures(tmp_path: Path) -> None:
    figures = make_figures()

    figure_paths = export_report_figures(
        figures=figures,
        output_dir=tmp_path,
        dpi=80,
    )

    assert set(figure_paths) == {"cumulative_returns", "drawdowns"}

    for path in figure_paths.values():
        assert path.exists()
        assert path.suffix == ".png"

    for figure in figures.values():
        plt.close(figure)


def test_write_report_markdown_index(tmp_path: Path) -> None:
    table_paths = export_report_tables(
        tables=make_tables(),
        output_dir=tmp_path,
    )
    figures = make_figures()
    figure_paths = export_report_figures(
        figures=figures,
        output_dir=tmp_path,
    )

    markdown_path = write_report_markdown_index(
        table_paths=table_paths,
        figure_paths=figure_paths,
        output_dir=tmp_path,
        title="Test Report",
    )

    assert markdown_path.exists()

    content = markdown_path.read_text(encoding="utf-8")

    assert "# Test Report" in content
    assert "strategy_metrics" in content
    assert "cumulative_returns" in content

    for figure in figures.values():
        plt.close(figure)


def test_write_report_manifest(tmp_path: Path) -> None:
    table_paths = export_report_tables(
        tables=make_tables(),
        output_dir=tmp_path,
    )
    figures = make_figures()
    figure_paths = export_report_figures(
        figures=figures,
        output_dir=tmp_path,
    )
    markdown_path = write_report_markdown_index(
        table_paths=table_paths,
        figure_paths=figure_paths,
        output_dir=tmp_path,
    )

    manifest_path = write_report_manifest(
        table_paths=table_paths,
        figure_paths=figure_paths,
        markdown_path=markdown_path,
        output_dir=tmp_path,
    )

    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "tables" in manifest
    assert "figures" in manifest
    assert manifest["tables"]["strategy_metrics"] == "tables/strategy_metrics.csv"

    for figure in figures.values():
        plt.close(figure)


def test_export_report_bundle(tmp_path: Path) -> None:
    figures = make_figures()

    result = export_report_bundle(
        tables=make_tables(),
        figures=figures,
        output_dir=tmp_path,
        title="Bundle Report",
        dpi=80,
    )

    assert isinstance(result, ReportExportResult)
    assert result.output_dir == tmp_path.resolve()
    assert result.markdown_path.exists()
    assert result.manifest_path.exists()
    assert set(result.table_paths) == {"strategy_metrics", "metric_deltas"}
    assert set(result.figure_paths) == {"cumulative_returns", "drawdowns"}

    for figure in figures.values():
        plt.close(figure)


def test_export_report_tables_rejects_empty_mapping(tmp_path: Path) -> None:
    with pytest.raises(ReportExportError, match="At least one table"):
        export_report_tables(
            tables={},
            output_dir=tmp_path,
        )


def test_export_report_tables_rejects_empty_table(tmp_path: Path) -> None:
    with pytest.raises(ReportExportError, match="Table cannot be empty"):
        export_report_tables(
            tables={"empty": pd.DataFrame()},
            output_dir=tmp_path,
        )


def test_export_report_figures_rejects_empty_mapping(tmp_path: Path) -> None:
    with pytest.raises(ReportExportError, match="At least one figure"):
        export_report_figures(
            figures={},
            output_dir=tmp_path,
        )


def test_export_report_figures_rejects_invalid_dpi(tmp_path: Path) -> None:
    figure = make_figure()

    with pytest.raises(ReportExportError, match="DPI"):
        export_report_figures(
            figures={"figure": figure},
            output_dir=tmp_path,
            dpi=0,
        )

    plt.close(figure)


def test_export_report_tables_rejects_invalid_artifact_name(tmp_path: Path) -> None:
    with pytest.raises(ReportExportError, match="Invalid artifact"):
        export_report_tables(
            tables={"bad/name": pd.DataFrame({"value": [1]})},
            output_dir=tmp_path,
        )


def test_output_dir_rejects_existing_file(tmp_path: Path) -> None:
    output_file = tmp_path / "not_a_directory"
    output_file.write_text("content", encoding="utf-8")

    with pytest.raises(ReportExportError, match="not a directory"):
        export_report_tables(
            tables=make_tables(),
            output_dir=output_file,
        )


def test_write_report_markdown_index_rejects_empty_title(tmp_path: Path) -> None:
    with pytest.raises(ReportExportError, match="title"):
        write_report_markdown_index(
            table_paths={},
            figure_paths={},
            output_dir=tmp_path,
            title=" ",
        )
