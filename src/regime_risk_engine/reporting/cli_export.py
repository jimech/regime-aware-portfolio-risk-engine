import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from regime_risk_engine.reporting.export import (
    ReportExportError,
    ReportExportResult,
    export_report_tables,
    write_report_manifest,
    write_report_markdown_index,
)


class CliReportExportError(ValueError):
    """Raised when CLI report export inputs are invalid."""


def export_report_from_files(
    table_specs: Sequence[str],
    figure_specs: Sequence[str],
    output_dir: Path | str,
    title: str = "Regime Risk Engine Report",
) -> ReportExportResult:
    """Export a report from named CSV table paths and optional PNG figure paths."""
    clean_output_dir = Path(output_dir).expanduser().resolve()
    tables = _load_tables_from_specs(table_specs)
    figure_paths = _copy_figures_from_specs(
        figure_specs=figure_specs,
        output_dir=clean_output_dir,
    )

    try:
        table_paths = export_report_tables(
            tables=tables,
            output_dir=clean_output_dir,
        )
        markdown_path = write_report_markdown_index(
            table_paths=table_paths,
            figure_paths=figure_paths,
            output_dir=clean_output_dir,
            title=title,
        )
        manifest_path = write_report_manifest(
            table_paths=table_paths,
            figure_paths=figure_paths,
            markdown_path=markdown_path,
            output_dir=clean_output_dir,
        )
    except ReportExportError as error:
        raise CliReportExportError(str(error)) from error

    return ReportExportResult(
        output_dir=clean_output_dir,
        table_paths=table_paths,
        figure_paths=figure_paths,
        markdown_path=markdown_path,
        manifest_path=manifest_path,
    )


def _load_tables_from_specs(table_specs: Sequence[str]) -> dict[str, pd.DataFrame]:
    if not table_specs:
        raise CliReportExportError("At least one --table argument is required")

    tables: dict[str, pd.DataFrame] = {}

    for table_spec in table_specs:
        table_name, table_path = _parse_named_path(table_spec)
        clean_table_name = _clean_artifact_name(table_name)

        if clean_table_name in tables:
            raise CliReportExportError(f"Duplicate table name: {clean_table_name}")

        if table_path.suffix.lower() != ".csv":
            raise CliReportExportError(f"Table file must be a CSV file: {table_path}")

        try:
            table = pd.read_csv(table_path)
        except pd.errors.EmptyDataError as error:
            raise CliReportExportError(f"Table CSV is empty: {table_path}") from error

        if table.empty:
            raise CliReportExportError(f"Table CSV has no rows: {table_path}")

        tables[clean_table_name] = table

    return tables


def _copy_figures_from_specs(
    figure_specs: Sequence[str],
    output_dir: Path,
) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}

    if not figure_specs:
        return figure_paths

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for figure_spec in figure_specs:
        figure_name, figure_path = _parse_named_path(figure_spec)
        clean_figure_name = _clean_artifact_name(figure_name)

        if clean_figure_name in figure_paths:
            raise CliReportExportError(f"Duplicate figure name: {clean_figure_name}")

        if figure_path.suffix.lower() != ".png":
            raise CliReportExportError(f"Figure file must be a PNG file: {figure_path}")

        destination_path = figure_dir / f"{clean_figure_name}.png"
        shutil.copy2(figure_path, destination_path)
        figure_paths[clean_figure_name] = destination_path

    return figure_paths


def _parse_named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise CliReportExportError("Artifact arguments must use the format name=path")

    name, raw_path = spec.split("=", maxsplit=1)
    clean_name = name.strip()
    clean_raw_path = raw_path.strip()

    if not clean_name:
        raise CliReportExportError("Artifact name must be non-empty")

    if not clean_raw_path:
        raise CliReportExportError("Artifact path must be non-empty")

    path = Path(clean_raw_path).expanduser().resolve()

    if not path.exists():
        raise CliReportExportError(f"Artifact path does not exist: {path}")

    if not path.is_file():
        raise CliReportExportError(f"Artifact path is not a file: {path}")

    return clean_name, path


def _clean_artifact_name(name: str) -> str:
    clean_name = str(name).strip().lower().replace(" ", "_")

    if not clean_name:
        raise CliReportExportError("Artifact names must be non-empty")

    invalid_characters = {"/", "\\", ":", "*", "?", '"', "<", ">", "|"}

    if any(character in clean_name for character in invalid_characters):
        raise CliReportExportError(f"Invalid artifact name: {name}")

    return clean_name
