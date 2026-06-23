from dataclasses import dataclass
from pathlib import Path

from regime_risk_engine.research.advanced_cli_export import (
    export_advanced_research_from_files,
)
from regime_risk_engine.research.advanced_demo import (
    AdvancedResearchDemoInputResult,
    StressPeriodMode,
    create_advanced_research_demo_inputs,
)
from regime_risk_engine.research.advanced_export import AdvancedResearchExportResult


class AdvancedResearchDemoWorkflowError(ValueError):
    """Raised when the advanced demo workflow cannot be completed."""


@dataclass(frozen=True, slots=True)
class AdvancedResearchDemoWorkflowResult:
    """Result of the one-command advanced demo workflow."""

    output_dir: Path
    input_result: AdvancedResearchDemoInputResult
    export_result: AdvancedResearchExportResult


def run_advanced_research_demo_workflow(
    output_dir: str | Path,
    n_regimes: int = 3,
    feature_window: int = 10,
    transaction_cost_bps: float = 5.0,
    random_state: int = 42,
    scenario_horizon: int = 21,
    scenario_simulations: int = 1_000,
    rolling_factor_window: int = 20,
    analyst: str | None = None,
    stress_period_mode: StressPeriodMode = "crisis",
    overwrite: bool = True,
) -> AdvancedResearchDemoWorkflowResult:
    """Create demo inputs and export a full advanced research package."""
    clean_output_dir = _prepare_output_dir(output_dir)

    input_dir = clean_output_dir / "inputs"
    package_dir = clean_output_dir / "package"

    input_result = create_advanced_research_demo_inputs(
        output_dir=input_dir,
        overwrite=overwrite,
        stress_period_mode=stress_period_mode,
    )

    export_result = export_advanced_research_from_files(
        price_data_path=input_result.price_data_path,
        static_weights_path=input_result.static_weights_path,
        regime_policy_path=input_result.regime_policy_path,
        stress_periods_path=input_result.stress_periods_path,
        factor_returns_path=input_result.factor_returns_path,
        output_dir=package_dir,
        n_regimes=n_regimes,
        feature_window=feature_window,
        transaction_cost_bps=transaction_cost_bps,
        random_state=random_state,
        scenario_horizon=scenario_horizon,
        scenario_simulations=scenario_simulations,
        rolling_factor_window=rolling_factor_window,
        analyst=analyst,
        overwrite=overwrite,
    )

    return AdvancedResearchDemoWorkflowResult(
        output_dir=clean_output_dir,
        input_result=input_result,
        export_result=export_result,
    )


def format_advanced_demo_workflow_result(
    result: AdvancedResearchDemoWorkflowResult,
) -> str:
    """Format the one-command advanced demo workflow result."""
    lines = [
        "Advanced demo workflow completed successfully.",
        f"Output directory: {result.output_dir}",
        f"Input directory: {result.input_result.output_dir}",
        f"Research package directory: {result.export_result.output_dir}",
        f"Advanced memo: {result.export_result.memo_path}",
    ]

    if result.export_result.exported_table_paths:
        lines.append("Exported tables:")

        for name, path in sorted(result.export_result.exported_table_paths.items()):
            lines.append(f"- {name}: {path}")

    return "\n".join(lines)


def _prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser().resolve()

    if path.exists() and not path.is_dir():
        raise AdvancedResearchDemoWorkflowError(
            f"Output path exists and is not a directory: {path}"
        )

    path.mkdir(parents=True, exist_ok=True)

    return path
