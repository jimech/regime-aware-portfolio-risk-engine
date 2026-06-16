from dataclasses import dataclass
from pathlib import Path

from regime_risk_engine.reporting.cli_export import (
    CliReportExportError,
    export_report_from_files,
)
from regime_risk_engine.reporting.demo import (
    DemoReportInputError,
    create_demo_report_inputs,
)


class DemoReportWorkflowError(ValueError):
    """Raised when the demo report workflow cannot be completed."""


@dataclass(frozen=True, slots=True)
class DemoReportWorkflowResult:
    """Container for generated demo report workflow paths."""

    output_dir: Path
    input_dir: Path
    report_dir: Path
    table_paths: dict[str, Path]
    figure_paths: dict[str, Path]
    markdown_path: Path
    manifest_path: Path

    def to_dict(self) -> dict[str, object]:
        """Convert the workflow result to a JSON-safe dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "input_dir": str(self.input_dir),
            "report_dir": str(self.report_dir),
            "tables": {
                name: str(path) for name, path in sorted(self.table_paths.items())
            },
            "figures": {
                name: str(path) for name, path in sorted(self.figure_paths.items())
            },
            "markdown_path": str(self.markdown_path),
            "manifest_path": str(self.manifest_path),
        }


def run_demo_report_workflow(
    output_dir: Path | str,
    title: str = "Demo Regime Risk Engine Report",
) -> DemoReportWorkflowResult:
    """Create demo inputs and export a demo report in one workflow."""
    clean_output_dir = _prepare_output_dir(output_dir)
    clean_title = str(title).strip()

    if not clean_title:
        raise DemoReportWorkflowError("Demo report title must be non-empty")

    input_dir = clean_output_dir / "inputs"
    report_dir = clean_output_dir / "report"

    try:
        demo_inputs = create_demo_report_inputs(output_dir=input_dir)
    except DemoReportInputError as error:
        raise DemoReportWorkflowError(str(error)) from error

    table_specs = [
        f"{name}={path}" for name, path in sorted(demo_inputs.table_paths.items())
    ]
    figure_specs = [
        f"{name}={path}" for name, path in sorted(demo_inputs.figure_paths.items())
    ]

    try:
        export_result = export_report_from_files(
            table_specs=table_specs,
            figure_specs=figure_specs,
            output_dir=report_dir,
            title=clean_title,
        )
    except CliReportExportError as error:
        raise DemoReportWorkflowError(str(error)) from error

    return DemoReportWorkflowResult(
        output_dir=clean_output_dir,
        input_dir=input_dir,
        report_dir=report_dir,
        table_paths=export_result.table_paths,
        figure_paths=export_result.figure_paths,
        markdown_path=export_result.markdown_path,
        manifest_path=export_result.manifest_path,
    )


def _prepare_output_dir(output_dir: Path | str) -> Path:
    clean_output_dir = Path(output_dir).expanduser().resolve()

    if clean_output_dir.exists() and not clean_output_dir.is_dir():
        raise DemoReportWorkflowError("Output path exists and is not a directory")

    clean_output_dir.mkdir(parents=True, exist_ok=True)

    return clean_output_dir
