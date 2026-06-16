import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


class ReportExportError(ValueError):
    """Raised when report exports cannot be created."""


@dataclass(frozen=True, slots=True)
class ReportExportResult:
    """Container for exported report artifact paths."""

    output_dir: Path
    table_paths: dict[str, Path]
    figure_paths: dict[str, Path]
    markdown_path: Path
    manifest_path: Path


def export_report_tables(
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path | str,
) -> dict[str, Path]:
    """Export report tables as CSV files."""
    clean_tables = _validate_table_mapping(tables)
    clean_output_dir = _prepare_output_dir(output_dir)

    table_dir = clean_output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    table_paths: dict[str, Path] = {}

    for table_name, table in clean_tables.items():
        path = table_dir / f"{table_name}.csv"
        table.to_csv(path, index=True)
        table_paths[table_name] = path

    return table_paths


def export_report_figures(
    figures: Mapping[str, Figure],
    output_dir: Path | str,
    dpi: int = 150,
) -> dict[str, Path]:
    """Export report figures as PNG files."""
    clean_figures = _validate_figure_mapping(figures)
    clean_output_dir = _prepare_output_dir(output_dir)
    _validate_dpi(dpi)

    figure_dir = clean_output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict[str, Path] = {}

    for figure_name, figure in clean_figures.items():
        path = figure_dir / f"{figure_name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        figure_paths[figure_name] = path

    return figure_paths


def write_report_markdown_index(
    table_paths: Mapping[str, Path],
    figure_paths: Mapping[str, Path],
    output_dir: Path | str,
    title: str = "Regime Risk Engine Report",
) -> Path:
    """Write a Markdown index linking exported tables and figures."""
    clean_output_dir = _prepare_output_dir(output_dir)
    clean_title = str(title).strip()

    if not clean_title:
        raise ReportExportError("Report title must be non-empty")

    markdown_path = clean_output_dir / "report_index.md"

    lines = [
        f"# {clean_title}",
        "",
        "## Tables",
        "",
    ]

    if table_paths:
        for table_name, path in sorted(table_paths.items()):
            lines.append(
                f"- **{table_name}**: `{_relative_path(path, clean_output_dir)}`"
            )
    else:
        lines.append("- No tables exported.")

    lines.extend(
        [
            "",
            "## Figures",
            "",
        ]
    )

    if figure_paths:
        for figure_name, path in sorted(figure_paths.items()):
            relative = _relative_path(path, clean_output_dir)
            lines.append(f"### {figure_name}")
            lines.append("")
            lines.append(f"![{figure_name}]({relative})")
            lines.append("")
    else:
        lines.append("- No figures exported.")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return markdown_path


def write_report_manifest(
    table_paths: Mapping[str, Path],
    figure_paths: Mapping[str, Path],
    markdown_path: Path,
    output_dir: Path | str,
) -> Path:
    """Write JSON manifest of exported report artifacts."""
    clean_output_dir = _prepare_output_dir(output_dir)

    manifest_path = clean_output_dir / "manifest.json"

    manifest = {
        "output_dir": str(clean_output_dir),
        "markdown_path": str(_relative_path(markdown_path, clean_output_dir)),
        "tables": {
            table_name: str(_relative_path(path, clean_output_dir))
            for table_name, path in sorted(table_paths.items())
        },
        "figures": {
            figure_name: str(_relative_path(path, clean_output_dir))
            for figure_name, path in sorted(figure_paths.items())
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest_path


def export_report_bundle(
    tables: Mapping[str, pd.DataFrame],
    figures: Mapping[str, Figure],
    output_dir: Path | str,
    title: str = "Regime Risk Engine Report",
    dpi: int = 150,
) -> ReportExportResult:
    """Export tables, figures, Markdown index, and manifest."""
    clean_output_dir = _prepare_output_dir(output_dir)

    table_paths = export_report_tables(
        tables=tables,
        output_dir=clean_output_dir,
    )
    figure_paths = export_report_figures(
        figures=figures,
        output_dir=clean_output_dir,
        dpi=dpi,
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

    return ReportExportResult(
        output_dir=clean_output_dir,
        table_paths=table_paths,
        figure_paths=figure_paths,
        markdown_path=markdown_path,
        manifest_path=manifest_path,
    )


def _validate_table_mapping(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    if not tables:
        raise ReportExportError("At least one table is required")

    clean_tables: dict[str, pd.DataFrame] = {}

    for table_name, table in tables.items():
        clean_name = _clean_artifact_name(table_name)

        if clean_name in clean_tables:
            raise ReportExportError(f"Duplicate table name: {clean_name}")

        if table.empty:
            raise ReportExportError(f"Table cannot be empty: {clean_name}")

        clean_tables[clean_name] = table.copy()

    return clean_tables


def _validate_figure_mapping(
    figures: Mapping[str, Figure],
) -> dict[str, Figure]:
    if not figures:
        raise ReportExportError("At least one figure is required")

    clean_figures: dict[str, Figure] = {}

    for figure_name, figure in figures.items():
        clean_name = _clean_artifact_name(figure_name)

        if clean_name in clean_figures:
            raise ReportExportError(f"Duplicate figure name: {clean_name}")

        if not isinstance(figure, Figure):
            raise ReportExportError(f"Object is not a matplotlib Figure: {clean_name}")

        clean_figures[clean_name] = figure

    return clean_figures


def _prepare_output_dir(output_dir: Path | str) -> Path:
    clean_output_dir = Path(output_dir).expanduser().resolve()

    if clean_output_dir.exists() and not clean_output_dir.is_dir():
        raise ReportExportError("Output path exists and is not a directory")

    clean_output_dir.mkdir(parents=True, exist_ok=True)

    return clean_output_dir


def _validate_dpi(dpi: int) -> None:
    if dpi <= 0:
        raise ReportExportError("Figure DPI must be positive")


def _clean_artifact_name(name: str) -> str:
    clean_name = str(name).strip().lower().replace(" ", "_")

    if not clean_name:
        raise ReportExportError("Artifact names must be non-empty")

    invalid_characters = {"/", "\\", ":", "*", "?", '"', "<", ">", "|"}

    if any(character in clean_name for character in invalid_characters):
        raise ReportExportError(f"Invalid artifact name: {name}")

    return clean_name


def _relative_path(path: Path, base_dir: Path) -> Path:
    return Path(path).resolve().relative_to(base_dir.resolve())
