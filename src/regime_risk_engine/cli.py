import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from regime_risk_engine import __version__
from regime_risk_engine.cli_config import (
    CliConfigInspectionError,
    format_config_inspection,
    inspect_config_file,
)
from regime_risk_engine.reporting.cli_export import (
    CliReportExportError,
    export_report_from_files,
)
from regime_risk_engine.reporting.demo import (
    DemoReportInputError,
    create_demo_report_inputs,
)
from regime_risk_engine.reporting.demo_workflow import (
    DemoReportWorkflowError,
    run_demo_report_workflow,
)
from regime_risk_engine.research.advanced_cli_export import (
    AdvancedResearchCliExportError,
    export_advanced_research_from_files,
    format_advanced_export_result,
)
from regime_risk_engine.research.advanced_demo import (
    AdvancedResearchDemoInputError,
    create_advanced_research_demo_inputs,
    format_advanced_demo_input_result,
)
from regime_risk_engine.research.advanced_demo_workflow import (
    AdvancedResearchDemoWorkflowError,
    format_advanced_demo_workflow_result,
    run_advanced_research_demo_workflow,
)


class CliError(ValueError):
    """Raised when CLI helper operations fail."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regime-risk-engine",
        description="Regime-aware portfolio risk engine command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser(
        "version",
        help="Show package version.",
    )
    version_parser.set_defaults(handler=_handle_version)

    healthcheck_parser = subparsers.add_parser(
        "healthcheck",
        help="Run a basic CLI healthcheck.",
    )
    healthcheck_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit healthcheck output as JSON.",
    )
    healthcheck_parser.add_argument(
        "--output-dir",
        help="Optional directory to create for CLI artifacts.",
    )
    healthcheck_parser.set_defaults(handler=_handle_healthcheck)

    inspect_config_parser = subparsers.add_parser(
        "inspect-config",
        help="Inspect a project configuration file.",
    )
    inspect_config_parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    inspect_config_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit config inspection result as JSON.",
    )
    inspect_config_parser.set_defaults(
        handler=_handle_inspect_config,
        parser=inspect_config_parser,
    )

    export_report_parser = subparsers.add_parser(
        "export-report",
        help="Export a Markdown research report from CSV inputs.",
    )
    export_report_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the report package should be written.",
    )
    export_report_parser.add_argument(
        "--table",
        action="append",
        default=[],
        help="Named CSV table path in the form name=path. May be repeated.",
    )
    export_report_parser.add_argument(
        "--figure",
        action="append",
        default=[],
        help="Named PNG figure path in the form name=path. May be repeated.",
    )
    export_report_parser.add_argument(
        "--title",
        default="Regime-Aware Portfolio Risk Report",
        help="Report title.",
    )
    export_report_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit report export result as JSON.",
    )
    export_report_parser.set_defaults(
        handler=_handle_export_report,
        parser=export_report_parser,
    )

    demo_inputs_parser = subparsers.add_parser(
        "create-demo-report-inputs",
        help="Create synthetic demo CSV inputs for report export.",
    )
    demo_inputs_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where demo input CSV files should be written.",
    )
    demo_inputs_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit created input paths as JSON.",
    )
    demo_inputs_parser.set_defaults(handler=_handle_create_demo_report_inputs)

    demo_report_parser = subparsers.add_parser(
        "run-demo-report",
        help="Create demo report inputs and export a demo Markdown report.",
    )
    demo_report_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where demo files should be written.",
    )
    demo_report_parser.add_argument(
        "--title",
        default="Regime-Aware Portfolio Risk Demo Report",
        help="Demo report title.",
    )
    demo_report_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit demo workflow result as JSON.",
    )
    demo_report_parser.set_defaults(
        handler=_handle_run_demo_report,
        parser=demo_report_parser,
    )

    advanced_export_parser = subparsers.add_parser(
        "export-advanced-research",
        help="Run the advanced research workflow and export a research package.",
    )
    advanced_export_parser.add_argument(
        "--price-data",
        required=True,
        help="CSV with date, ticker, and adjusted_close columns.",
    )
    advanced_export_parser.add_argument(
        "--static-weights",
        required=True,
        help="CSV with ticker and weight columns.",
    )
    advanced_export_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the advanced research package should be written.",
    )
    advanced_export_parser.add_argument(
        "--regime-policy",
        help="Optional CSV with regime, ticker, and weight columns.",
    )
    advanced_export_parser.add_argument(
        "--stress-periods",
        help="Optional CSV with name, start_date, and end_date columns.",
    )
    advanced_export_parser.add_argument(
        "--factor-returns",
        help="Optional CSV with date plus one or more factor return columns.",
    )
    advanced_export_parser.add_argument(
        "--n-regimes",
        type=int,
        default=3,
        help="Number of regimes to detect.",
    )
    advanced_export_parser.add_argument(
        "--feature-window",
        type=int,
        default=21,
        help="Rolling feature window used for regime detection.",
    )
    advanced_export_parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=5.0,
        help="Transaction cost assumption in basis points.",
    )
    advanced_export_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible modeling and simulation.",
    )
    advanced_export_parser.add_argument(
        "--scenario-horizon",
        type=int,
        default=21,
        help="Forward scenario simulation horizon.",
    )
    advanced_export_parser.add_argument(
        "--scenario-simulations",
        type=int,
        default=1_000,
        help="Number of forward scenario simulations.",
    )
    advanced_export_parser.add_argument(
        "--analyst",
        help="Optional analyst name.",
    )
    advanced_export_parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if export files already exist.",
    )
    advanced_export_parser.set_defaults(
        handler=_handle_export_advanced_research,
        parser=advanced_export_parser,
    )

    advanced_demo_parser = subparsers.add_parser(
        "create-advanced-demo-inputs",
        help="Create demo CSV inputs for the advanced research workflow.",
    )
    advanced_demo_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where advanced demo input CSV files should be written.",
    )
    advanced_demo_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit created advanced demo input paths as JSON.",
    )
    advanced_demo_parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if demo input files already exist.",
    )
    advanced_demo_parser.add_argument(
        "--stress-period-mode",
        choices=["crisis", "synthetic"],
        default="crisis",
        help=(
            "Stress-period source for generated demo inputs. "
            "Use 'crisis' for historical crisis presets or 'synthetic' "
            "for a deterministic demo-only window."
        ),
    )
    advanced_demo_parser.set_defaults(
        handler=_handle_create_advanced_demo_inputs,
        parser=advanced_demo_parser,
    )

    advanced_demo_workflow_parser = subparsers.add_parser(
        "run-advanced-demo",
        help="Create demo inputs and export a full advanced research package.",
    )
    advanced_demo_workflow_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where advanced demo outputs should be written.",
    )
    advanced_demo_workflow_parser.add_argument(
        "--n-regimes",
        type=int,
        default=3,
        help="Number of regimes to detect.",
    )
    advanced_demo_workflow_parser.add_argument(
        "--feature-window",
        type=int,
        default=10,
        help="Rolling feature window used for regime detection.",
    )
    advanced_demo_workflow_parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=5.0,
        help="Transaction cost assumption in basis points.",
    )
    advanced_demo_workflow_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible modeling and simulation.",
    )
    advanced_demo_workflow_parser.add_argument(
        "--scenario-horizon",
        type=int,
        default=21,
        help="Forward scenario simulation horizon.",
    )
    advanced_demo_workflow_parser.add_argument(
        "--scenario-simulations",
        type=int,
        default=1_000,
        help="Number of forward scenario simulations.",
    )
    advanced_demo_workflow_parser.add_argument(
        "--analyst",
        help="Optional analyst name.",
    )
    advanced_demo_workflow_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit advanced demo workflow result as JSON.",
    )
    advanced_demo_workflow_parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if demo output files already exist.",
    )
    advanced_demo_workflow_parser.add_argument(
        "--stress-period-mode",
        choices=["crisis", "synthetic"],
        default="crisis",
        help=(
            "Stress-period source for generated demo inputs. "
            "Use 'crisis' for historical crisis presets or 'synthetic' "
            "for a deterministic demo-only window."
        ),
    )
    advanced_demo_workflow_parser.set_defaults(
        handler=_handle_run_advanced_demo,
        parser=advanced_demo_workflow_parser,
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler

    typed_handler: Callable[[argparse.Namespace], int] = handler

    return typed_handler(args)


def _handle_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _handle_healthcheck(_args: argparse.Namespace) -> int:
    result = build_healthcheck_result(output_dir=_args.output_dir)

    if _args.json:
        print(json.dumps(result, sort_keys=True))
        return 0

    print("Regime Risk Engine healthcheck")

    for key, value in result.items():
        print(f"{key}: {value}")

    return 0


def build_healthcheck_result(
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Build a JSON-safe CLI healthcheck result."""
    result: dict[str, object] = {
        "package_version": __version__,
        "import_ok": True,
        "output_dir": None,
    }

    if output_dir is None:
        return result

    clean_output_dir = Path(output_dir).expanduser().resolve()

    if clean_output_dir.exists() and not clean_output_dir.is_dir():
        raise CliError("Output path exists and is not a directory")

    clean_output_dir.mkdir(parents=True, exist_ok=True)

    result["output_dir"] = str(clean_output_dir)
    result["output_dir_exists"] = clean_output_dir.exists()

    return result


def _handle_inspect_config(args: argparse.Namespace) -> int:
    try:
        result = inspect_config_file(args.config)
    except CliConfigInspectionError as exc:
        args.parser.error(str(exc))

    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True))
        return 0

    print(format_config_inspection(result))
    return 0


def _handle_export_report(args: argparse.Namespace) -> int:
    if not args.table:
        args.parser.error("At least one --table argument is required")

    try:
        result = export_report_from_files(
            table_specs=args.table,
            figure_specs=args.figure,
            output_dir=args.output_dir,
            title=args.title,
        )
    except CliReportExportError as exc:
        args.parser.error(str(exc))

    if args.json:
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "tables": {
                        name: str(path)
                        for name, path in sorted(result.table_paths.items())
                    },
                    "figures": {
                        name: str(path)
                        for name, path in sorted(result.figure_paths.items())
                    },
                    "markdown_path": str(result.markdown_path),
                    "manifest_path": str(result.manifest_path),
                },
                sort_keys=True,
            )
        )
        return 0

    print(f"Report exported successfully: {result.markdown_path}")
    return 0


def _handle_create_demo_report_inputs(args: argparse.Namespace) -> int:
    try:
        result = create_demo_report_inputs(args.output_dir)
    except DemoReportInputError as exc:
        print(f"Demo report input creation failed: {exc}")
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "tables": {
                        name: str(path)
                        for name, path in sorted(result.table_paths.items())
                    },
                    "figures": {
                        name: str(path)
                        for name, path in sorted(result.figure_paths.items())
                    },
                },
                sort_keys=True,
            )
        )
        return 0

    print("Demo report inputs created successfully.")
    print(f"Output directory: {result.output_dir}")

    for name, path in sorted(result.table_paths.items()):
        print(f"Table {name}: {path}")

    for name, path in sorted(result.figure_paths.items()):
        print(f"Figure {name}: {path}")

    return 0


def _handle_run_demo_report(args: argparse.Namespace) -> int:
    try:
        result = run_demo_report_workflow(
            output_dir=args.output_dir,
            title=args.title,
        )
    except DemoReportWorkflowError as exc:
        args.parser.error(str(exc))

    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True))
        return 0

    print("Demo report workflow completed successfully.")
    print(f"Output directory: {result.output_dir}")
    print(f"Report: {result.markdown_path}")

    for name, path in sorted(result.table_paths.items()):
        print(f"Table {name}: {path}")

    for name, path in sorted(result.figure_paths.items()):
        print(f"Figure {name}: {path}")

    return 0


def _handle_export_advanced_research(args: argparse.Namespace) -> int:
    try:
        result = export_advanced_research_from_files(
            price_data_path=args.price_data,
            static_weights_path=args.static_weights,
            regime_policy_path=args.regime_policy,
            stress_periods_path=args.stress_periods,
            factor_returns_path=args.factor_returns,
            output_dir=args.output_dir,
            n_regimes=args.n_regimes,
            feature_window=args.feature_window,
            transaction_cost_bps=args.transaction_cost_bps,
            random_state=args.random_state,
            scenario_horizon=args.scenario_horizon,
            scenario_simulations=args.scenario_simulations,
            analyst=args.analyst,
            overwrite=not args.no_overwrite,
        )
    except AdvancedResearchCliExportError as exc:
        print(f"Advanced research export failed: {exc}")
        return 1

    print(format_advanced_export_result(result))
    return 0


def _handle_create_advanced_demo_inputs(args: argparse.Namespace) -> int:
    try:
        result = create_advanced_research_demo_inputs(
            output_dir=args.output_dir,
            overwrite=not args.no_overwrite,
            stress_period_mode=args.stress_period_mode,
        )
    except AdvancedResearchDemoInputError as exc:
        print(f"Advanced demo input creation failed: {exc}")
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "price_data_path": str(result.price_data_path),
                    "static_weights_path": str(result.static_weights_path),
                    "regime_policy_path": str(result.regime_policy_path),
                    "stress_periods_path": str(result.stress_periods_path),
                    "factor_returns_path": str(result.factor_returns_path),
                },
                sort_keys=True,
            )
        )
        return 0

    print(format_advanced_demo_input_result(result))
    return 0


def _handle_run_advanced_demo(args: argparse.Namespace) -> int:
    try:
        result = run_advanced_research_demo_workflow(
            output_dir=args.output_dir,
            n_regimes=args.n_regimes,
            feature_window=args.feature_window,
            transaction_cost_bps=args.transaction_cost_bps,
            random_state=args.random_state,
            scenario_horizon=args.scenario_horizon,
            scenario_simulations=args.scenario_simulations,
            analyst=args.analyst,
            overwrite=not args.no_overwrite,
            stress_period_mode=args.stress_period_mode,
        )
    except (AdvancedResearchDemoWorkflowError, ValueError) as exc:
        print(f"Advanced demo workflow failed: {exc}")
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "input_dir": str(result.input_result.output_dir),
                    "package_dir": str(result.export_result.output_dir),
                    "memo_path": str(result.export_result.memo_path),
                    "exported_tables": {
                        name: str(path)
                        for name, path in sorted(
                            result.export_result.exported_table_paths.items()
                        )
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(format_advanced_demo_workflow_result(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
