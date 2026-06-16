import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from regime_risk_engine import __version__
from regime_risk_engine.reporting.cli_export import (
    CliReportExportError,
    export_report_from_files,
)
from regime_risk_engine.reporting.demo import (
    DemoReportInputError,
    create_demo_report_inputs,
)


class CliError(ValueError):
    """Raised when a CLI command cannot be executed."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="regime-risk-engine",
        description="Regime-aware portfolio risk engine command-line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"regime-risk-engine {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    version_parser = subparsers.add_parser(
        "version",
        help="Print the package version.",
    )
    version_parser.set_defaults(handler=_handle_version)

    healthcheck_parser = subparsers.add_parser(
        "healthcheck",
        help="Run a basic project healthcheck.",
    )
    healthcheck_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory to validate or create.",
    )
    healthcheck_parser.add_argument(
        "--json",
        action="store_true",
        help="Print healthcheck output as JSON.",
    )
    healthcheck_parser.set_defaults(handler=_handle_healthcheck)

    export_report_parser = subparsers.add_parser(
        "export-report",
        help="Export report-ready CSV tables and optional PNG figures.",
    )
    export_report_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where report artifacts will be written.",
    )
    export_report_parser.add_argument(
        "--table",
        action="append",
        default=[],
        help="Named CSV table input in the format name=path.",
    )
    export_report_parser.add_argument(
        "--figure",
        action="append",
        default=[],
        help="Named PNG figure input in the format name=path.",
    )
    export_report_parser.add_argument(
        "--title",
        default="Regime Risk Engine Report",
        help="Report title used in the Markdown index.",
    )
    export_report_parser.add_argument(
        "--json",
        action="store_true",
        help="Print export result as JSON.",
    )
    export_report_parser.set_defaults(handler=_handle_export_report)

    demo_parser = subparsers.add_parser(
        "create-demo-report-inputs",
        help="Create demo CSV tables and PNG figures for report export.",
    )
    demo_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where demo report inputs will be written.",
    )
    demo_parser.add_argument(
        "--json",
        action="store_true",
        help="Print generated paths as JSON.",
    )
    demo_parser.set_defaults(handler=_handle_create_demo_report_inputs)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)

    if handler is None:
        parser.print_help()
        return 1

    try:
        return int(handler(args))
    except CliError as error:
        parser.exit(status=2, message=f"error: {error}\n")


def _handle_version(args: argparse.Namespace) -> int:
    _ = args

    print(__version__)

    return 0


def _handle_healthcheck(args: argparse.Namespace) -> int:
    output_dir = getattr(args, "output_dir", None)
    as_json = bool(getattr(args, "json", False))

    result = build_healthcheck_result(output_dir=output_dir)

    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Regime Risk Engine healthcheck")
        print(f"- package_version: {result['package_version']}")
        print(f"- import_ok: {result['import_ok']}")
        print(f"- working_directory: {result['working_directory']}")

        if result["output_dir"] is not None:
            print(f"- output_dir: {result['output_dir']}")
            print(f"- output_dir_exists: {result['output_dir_exists']}")

    return 0


def _handle_export_report(args: argparse.Namespace) -> int:
    table_specs = list(getattr(args, "table", []))
    figure_specs = list(getattr(args, "figure", []))
    output_dir = args.output_dir
    title = str(getattr(args, "title", "Regime Risk Engine Report"))
    as_json = bool(getattr(args, "json", False))

    try:
        result = export_report_from_files(
            table_specs=table_specs,
            figure_specs=figure_specs,
            output_dir=output_dir,
            title=title,
        )
    except CliReportExportError as error:
        raise CliError(str(error)) from error

    export_result = {
        "output_dir": str(result.output_dir),
        "markdown_path": str(result.markdown_path),
        "manifest_path": str(result.manifest_path),
        "tables": {
            name: str(path) for name, path in sorted(result.table_paths.items())
        },
        "figures": {
            name: str(path) for name, path in sorted(result.figure_paths.items())
        },
    }

    if as_json:
        print(json.dumps(export_result, indent=2, sort_keys=True))
    else:
        print("Report exported")
        print(f"- output_dir: {result.output_dir}")
        print(f"- markdown_path: {result.markdown_path}")
        print(f"- manifest_path: {result.manifest_path}")
        print(f"- tables: {len(result.table_paths)}")
        print(f"- figures: {len(result.figure_paths)}")

    return 0


def _handle_create_demo_report_inputs(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    as_json = bool(getattr(args, "json", False))

    try:
        result = create_demo_report_inputs(output_dir=output_dir)
    except DemoReportInputError as error:
        raise CliError(str(error)) from error

    demo_result = {
        "output_dir": str(result.output_dir),
        "tables": {
            name: str(path) for name, path in sorted(result.table_paths.items())
        },
        "figures": {
            name: str(path) for name, path in sorted(result.figure_paths.items())
        },
    }

    if as_json:
        print(json.dumps(demo_result, indent=2, sort_keys=True))
    else:
        print("Demo report inputs created")
        print(f"- output_dir: {result.output_dir}")
        print(f"- tables: {len(result.table_paths)}")
        print(f"- figures: {len(result.figure_paths)}")

    return 0


def build_healthcheck_result(output_dir: Path | None = None) -> dict[str, Any]:
    """Build a basic CLI healthcheck result."""
    result: dict[str, Any] = {
        "package_version": __version__,
        "import_ok": True,
        "working_directory": str(Path.cwd()),
        "output_dir": None,
        "output_dir_exists": None,
    }

    if output_dir is not None:
        clean_output_dir = output_dir.expanduser().resolve()

        if clean_output_dir.exists() and not clean_output_dir.is_dir():
            raise CliError("Output path exists and is not a directory")

        clean_output_dir.mkdir(parents=True, exist_ok=True)

        result["output_dir"] = str(clean_output_dir)
        result["output_dir_exists"] = clean_output_dir.exists()

    return result
