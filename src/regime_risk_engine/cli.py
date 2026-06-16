import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from regime_risk_engine import __version__


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
