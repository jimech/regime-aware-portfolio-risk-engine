from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.research.attribution import StrategyAttributionSummary
from regime_risk_engine.research.factor_exposure import FactorExposureSummary
from regime_risk_engine.research.factor_significance import FactorSignificanceResult
from regime_risk_engine.research.regime_intelligence import RegimeIntelligenceSummary
from regime_risk_engine.research.regime_transitions import RegimeTransitionSummary
from regime_risk_engine.research.rolling_factor_exposure import (
    RollingFactorExposureResult,
    summarize_rolling_factor_exposures,
)
from regime_risk_engine.research.scenario_simulation import (
    RegimeScenarioSimulationResult,
)
from regime_risk_engine.research.stress_testing import StressTestSummary


class AdvancedResearchMemoError(ValueError):
    """Raised when an advanced research memo cannot be created."""


@dataclass(frozen=True, slots=True)
class AdvancedResearchMemoConfig:
    """Configuration for advanced research memo generation."""

    title: str = "Advanced Regime-Aware Portfolio Research Memo"
    analyst: str | None = None
    include_limitations: bool = True


@dataclass(frozen=True, slots=True)
class AdvancedResearchMemoInputs:
    """Optional advanced research summaries to include in a memo."""

    base_memo: str
    regime_intelligence: RegimeIntelligenceSummary | None = None
    regime_transitions: RegimeTransitionSummary | None = None
    stress_test: StressTestSummary | None = None
    attribution: StrategyAttributionSummary | None = None
    factor_exposure: FactorExposureSummary | None = None
    rolling_factor_exposure: RollingFactorExposureResult | None = None
    scenario_simulation: RegimeScenarioSimulationResult | None = None
    factor_significance: FactorSignificanceResult | None = None


def build_advanced_research_memo(
    inputs: AdvancedResearchMemoInputs,
    config: AdvancedResearchMemoConfig | None = None,
) -> str:
    """Build an advanced Markdown investment research memo."""
    memo_config = config or AdvancedResearchMemoConfig()

    _validate_config(memo_config)
    _validate_inputs(inputs)

    sections = [
        _build_header(memo_config),
        _build_base_memo_section(inputs.base_memo),
    ]

    if inputs.regime_intelligence is not None:
        sections.append(_build_regime_intelligence_section(inputs.regime_intelligence))

    if inputs.regime_transitions is not None:
        sections.append(_build_regime_transition_section(inputs.regime_transitions))

    if inputs.stress_test is not None:
        sections.append(_build_stress_test_section(inputs.stress_test))

    if inputs.attribution is not None:
        sections.append(_build_attribution_section(inputs.attribution))

    if inputs.factor_exposure is not None:
        sections.append(_build_factor_exposure_section(inputs.factor_exposure))

    if inputs.rolling_factor_exposure is not None:
        sections.append(
            _build_rolling_factor_exposure_section(inputs.rolling_factor_exposure)
        )

    if inputs.factor_significance is not None:
        sections.append(_build_factor_significance_section(inputs.factor_significance))

    if inputs.scenario_simulation is not None:
        sections.append(_build_scenario_simulation_section(inputs.scenario_simulation))

    sections.append(_build_research_takeaway_section(inputs))

    if memo_config.include_limitations:
        sections.append(_build_limitations_section())

    return "\n\n".join(sections).strip() + "\n"


def _validate_config(config: AdvancedResearchMemoConfig) -> None:
    if not config.title.strip():
        raise AdvancedResearchMemoError("Memo title must be non-empty")

    if config.analyst is not None and not config.analyst.strip():
        raise AdvancedResearchMemoError("Analyst name must be non-empty when provided")


def _validate_inputs(inputs: AdvancedResearchMemoInputs) -> None:
    if not inputs.base_memo.strip():
        raise AdvancedResearchMemoError("Base memo must be non-empty")


def _build_header(config: AdvancedResearchMemoConfig) -> str:
    lines = [f"# {config.title}"]

    if config.analyst is not None:
        lines.append(f"**Analyst:** {config.analyst.strip()}")

    return "\n\n".join(lines)


def _build_base_memo_section(base_memo: str) -> str:
    return (
        "## Base Research Memo\n\n"
        "The following section summarizes the core market research workflow output.\n\n"
        f"{base_memo.strip()}"
    )


def _build_regime_intelligence_section(
    summary: RegimeIntelligenceSummary,
) -> str:
    rows = []

    for row in summary.profile_table.to_dict(orient="records"):
        rows.append(
            [
                str(int(row["regime"])),
                str(row["label"]),
                _format_percent(row["annualized_return"]),
                _format_percent(row["annualized_volatility"]),
                _format_percent(row["max_drawdown"]),
                str(row["best_asset"]),
                str(row["worst_asset"]),
            ]
        )

    table = _build_markdown_table(
        headers=[
            "Regime",
            "Label",
            "Ann. Return",
            "Ann. Volatility",
            "Max Drawdown",
            "Best Asset",
            "Worst Asset",
        ],
        rows=rows,
    )

    return f"## Regime Intelligence\n\n{summary.narrative}\n\n{table}"


def _build_regime_transition_section(
    summary: RegimeTransitionSummary,
) -> str:
    rows = []

    for row in summary.regime_persistence.to_dict(orient="records"):
        rows.append(
            [
                str(int(row["regime"])),
                _format_percent(row["persistence_probability"]),
                _format_decimal(row["expected_duration"]),
                str(int(row["transition_observation_count"])),
            ]
        )

    table = _build_markdown_table(
        headers=[
            "Regime",
            "Persistence Probability",
            "Expected Duration",
            "Transition Observations",
        ],
        rows=rows,
    )

    return f"## Regime Transition Risk\n\n{summary.narrative}\n\n{table}"


def _build_stress_test_section(summary: StressTestSummary) -> str:
    rows = []

    for row in summary.summary_table.to_dict(orient="records"):
        rows.append(
            [
                str(row["period_name"]),
                _format_percent(row["return_delta"]),
                _format_percent(row["drawdown_delta"]),
                _format_percent(row["volatility_delta"]),
                str(row["assessment"]),
            ]
        )

    table = _build_markdown_table(
        headers=[
            "Stress Period",
            "Return Delta",
            "Drawdown Delta",
            "Volatility Delta",
            "Assessment",
        ],
        rows=rows,
    )

    return f"## Stress-Period Analysis\n\n{summary.narrative}\n\n{table}"


def _build_attribution_section(summary: StrategyAttributionSummary) -> str:
    rows = []

    for row in summary.asset_attribution.to_dict(orient="records"):
        rows.append(
            [
                str(row["asset"]),
                _format_percent(row["average_active_weight"]),
                _format_percent(row["active_return_contribution"]),
            ]
        )

    table = _build_markdown_table(
        headers=[
            "Asset",
            "Average Active Weight",
            "Active Return Contribution",
        ],
        rows=rows,
    )

    return f"## Strategy Attribution\n\n{summary.narrative}\n\n{table}"


def _build_factor_exposure_section(summary: FactorExposureSummary) -> str:
    rows = []

    for row in summary.dominant_factor_by_strategy.to_dict(orient="records"):
        rows.append(
            [
                str(row["strategy"]),
                str(row["dominant_factor"]),
                _format_decimal(row["dominant_beta"]),
            ]
        )

    table = _build_markdown_table(
        headers=[
            "Strategy",
            "Dominant Factor",
            "Dominant Beta",
        ],
        rows=rows,
    )

    return f"## Factor Exposure Analysis\n\n{summary.narrative}\n\n{table}"


def _build_rolling_factor_exposure_section(
    summary: RollingFactorExposureResult,
) -> str:
    exposure_summary = summarize_rolling_factor_exposures(summary)

    rows = []

    for row in exposure_summary.to_dict(orient="records"):
        rows.append(
            [
                str(row["factor"]),
                _format_decimal(row["latest_beta"]),
                _format_decimal(row["average_beta"]),
                _format_decimal(row["minimum_beta"]),
                _format_decimal(row["maximum_beta"]),
                _format_decimal(row["beta_volatility"]),
            ]
        )

    table = _build_markdown_table(
        headers=[
            "Factor",
            "Latest Beta",
            "Average Beta",
            "Minimum Beta",
            "Maximum Beta",
            "Beta Volatility",
        ],
        rows=rows,
    )

    return (
        "## Rolling Factor Exposure Analysis\n\n"
        "Rolling factor exposure analysis estimates how the dynamic strategy's "
        "factor betas changed through time. This helps evaluate whether "
        "regime-aware allocation decisions translated into measurable changes "
        "in equity, defensive, or real-asset risk exposure.\n\n"
        f"{table}"
    )


def _build_factor_significance_section(
    summary: FactorSignificanceResult,
) -> str:
    rows = []

    for row in summary.significance_table.to_dict(orient="records"):
        rows.append(
            [
                str(row["strategy"]),
                str(row["factor"]),
                _format_decimal(row["beta"]),
                _format_decimal(row["standard_error"]),
                _format_decimal(row["t_stat"]),
                _format_decimal(row["p_value"]),
                "yes" if bool(row["significant"]) else "no",
            ]
        )

    table = _build_markdown_table(
        headers=[
            "Strategy",
            "Factor",
            "Beta",
            "Std. Error",
            "T-Stat",
            "P-Value",
            "Significant",
        ],
        rows=rows,
    )

    return (
        "## Factor Significance Analysis\n\n"
        "Factor significance analysis estimates whether the dynamic strategy's "
        "factor betas are statistically distinguishable from zero under an "
        "ordinary least squares diagnostic. This helps separate economically "
        "large exposures from noisy factor relationships.\n\n"
        f"Regression alpha: {_format_decimal(summary.alpha)}\n\n"
        f"Regression R-squared: {_format_decimal(summary.r_squared)}\n\n"
        f"Observations: {summary.observations}\n\n"
        f"{table}"
    )


def _build_scenario_simulation_section(
    summary: RegimeScenarioSimulationResult,
) -> str:
    rows = []

    for row in summary.terminal_summary.to_dict(orient="records"):
        rows.append(
            [
                str(row["strategy"]),
                _format_percent(row["mean_terminal_return"]),
                _format_percent(row["probability_of_loss"]),
                _format_percent(row["var_95"]),
                _format_percent(row["cvar_95"]),
            ]
        )

    table = _build_markdown_table(
        headers=[
            "Strategy",
            "Mean Terminal Return",
            "Probability of Loss",
            "VaR 95",
            "CVaR 95",
        ],
        rows=rows,
    )

    return f"## Forward Regime Scenario Simulation\n\n{summary.narrative}\n\n{table}"


def _build_research_takeaway_section(inputs: AdvancedResearchMemoInputs) -> str:
    available_sections = []

    if inputs.regime_intelligence is not None:
        available_sections.append("regime interpretation")

    if inputs.regime_transitions is not None:
        available_sections.append("transition stability")

    if inputs.stress_test is not None:
        available_sections.append("stress-period protection")

    if inputs.attribution is not None:
        available_sections.append("performance attribution")

    if inputs.factor_exposure is not None:
        available_sections.append("factor exposure diagnostics")

    if inputs.rolling_factor_exposure is not None:
        available_sections.append("rolling factor exposure diagnostics")

    if inputs.factor_significance is not None:
        available_sections.append("factor significance diagnostics")

    if inputs.scenario_simulation is not None:
        available_sections.append("forward scenario simulation")

    if not available_sections:
        return (
            "## Research Takeaway\n\n"
            "Only the base research memo was supplied. Add advanced research "
            "summaries to produce a fuller investment due-diligence memo."
        )

    section_text = ", ".join(available_sections)

    return (
        "## Research Takeaway\n\n"
        "This memo combines the base regime-aware portfolio analysis with "
        f"{section_text}. Together, these sections evaluate not only whether "
        "the dynamic strategy performed well, but also why it performed that "
        "way, how stable the regimes were, how the strategy behaved in stress, "
        "and what forward-looking risks remain."
    )


def _build_limitations_section() -> str:
    return (
        "## Advanced Research Limitations\n\n"
        "- Regime labels are estimated and may change under different model settings.\n"
        "- Optimized allocations can overfit if not evaluated out of sample.\n"
        "- Stress-period analysis depends on selected date windows.\n"
        "- Factor exposures depend on the supplied factor set.\n"
        "- Scenario simulations are not forecasts; they are regime-conditioned "
        "risk simulations based on historical behavior."
    )


def _build_markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]

    return "\n".join([header_line, separator_line, *row_lines])


def _format_percent(value: object) -> str:
    numeric = _to_float(value)

    return f"{numeric * 100:.2f}%"


def _format_decimal(value: object) -> str:
    numeric = _to_float(value)

    if pd.isna(numeric):
        return "n/a"

    if numeric == float("inf"):
        return "∞"

    return f"{numeric:.4f}"


def _to_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        return float("nan")

    return float(numeric)
