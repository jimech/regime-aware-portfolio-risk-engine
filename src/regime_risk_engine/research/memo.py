from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.research.market_workflow import MarketResearchWorkflowResult


class MarketResearchMemoError(ValueError):
    """Raised when a market research memo cannot be created."""


@dataclass(frozen=True, slots=True)
class MarketResearchMemoConfig:
    """Configuration for market research memo generation."""

    title: str = "Regime-Aware Portfolio Research Memo"
    analyst: str | None = None
    include_limitations: bool = True


PERCENT_METRICS = {
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "var",
    "cvar",
    "turnover",
    "transaction_cost",
    "transaction_costs",
}


def build_market_research_memo(
    result: MarketResearchWorkflowResult,
    config: MarketResearchMemoConfig | None = None,
) -> str:
    """Build a professional Markdown investment research memo."""
    memo_config = config or MarketResearchMemoConfig()

    _validate_config(memo_config)
    _validate_result(result)

    sections = [
        _build_header(memo_config),
        _build_executive_summary_section(result),
        _build_strategy_performance_section(result),
        _build_regime_findings_section(result),
        _build_allocation_profile_section(result),
        _build_research_conclusion_section(result),
    ]

    if memo_config.include_limitations:
        sections.append(_build_limitations_section())

    return "\n\n".join(sections).strip() + "\n"


def _validate_config(config: MarketResearchMemoConfig) -> None:
    if not config.title.strip():
        raise MarketResearchMemoError("Memo title must be non-empty")

    if config.analyst is not None and not config.analyst.strip():
        raise MarketResearchMemoError("Analyst name must be non-empty when provided")


def _validate_result(result: MarketResearchWorkflowResult) -> None:
    if result.asset_returns.empty:
        raise MarketResearchMemoError("Asset returns cannot be empty")

    if result.regime_labels.empty:
        raise MarketResearchMemoError("Regime labels cannot be empty")

    if result.research_result.metric_delta_table.empty:
        raise MarketResearchMemoError("Metric delta table cannot be empty")

    if result.research_result.regime_metric_table.empty:
        raise MarketResearchMemoError("Regime metric table cannot be empty")

    if not result.research_result.executive_summary.strip():
        raise MarketResearchMemoError("Executive summary cannot be empty")


def _build_header(config: MarketResearchMemoConfig) -> str:
    lines = [f"# {config.title}"]

    if config.analyst is not None:
        lines.append(f"**Analyst:** {config.analyst.strip()}")

    return "\n\n".join(lines)


def _build_executive_summary_section(result: MarketResearchWorkflowResult) -> str:
    start_date = result.asset_returns.index.min().date().isoformat()
    end_date = result.asset_returns.index.max().date().isoformat()
    observation_count = len(result.asset_returns)
    regime_count = int(result.regime_labels.nunique())

    return (
        "## Executive Summary\n\n"
        f"{result.research_result.executive_summary}\n\n"
        f"The analysis covers {observation_count} return observations from "
        f"{start_date} to {end_date}. The workflow identified {regime_count} "
        "market regimes and compared a static benchmark against a dynamic "
        "regime-aware allocation."
    )


def _build_strategy_performance_section(
    result: MarketResearchWorkflowResult,
) -> str:
    rows = []

    for row in result.research_result.metric_delta_table.to_dict(orient="records"):
        metric = str(row["metric"])
        metric_label = str(row.get("metric_label", metric))

        rows.append(
            [
                metric_label,
                _format_metric_value(metric, row["benchmark_value"]),
                _format_metric_value(metric, row["candidate_value"]),
                _format_metric_value(metric, row["absolute_delta"]),
                _format_relative_delta(row["relative_delta"]),
            ]
        )

    table = _build_markdown_table(
        headers=[
            "Metric",
            "Static Benchmark",
            "Dynamic Strategy",
            "Absolute Delta",
            "Relative Delta",
        ],
        rows=rows,
    )

    return (
        "## Strategy Performance\n\n"
        "The table below compares the dynamic regime-aware strategy against "
        "the static benchmark.\n\n"
        f"{table}"
    )


def _build_regime_findings_section(result: MarketResearchWorkflowResult) -> str:
    regime_summary = result.research_result.regime_summary.regime_summary
    rows = []

    for row in regime_summary.to_dict(orient="records"):
        is_favorable = bool(row["is_favorable"])
        rows.append(
            [
                str(int(row["regime"])),
                _format_decimal(row["benchmark_value"]),
                _format_decimal(row["candidate_value"]),
                _format_decimal(row["absolute_delta"]),
                "Favorable" if is_favorable else "Unfavorable",
            ]
        )

    table = _build_markdown_table(
        headers=[
            "Regime",
            "Static Metric",
            "Dynamic Metric",
            "Delta",
            "Assessment",
        ],
        rows=rows,
    )

    return (
        "## Regime-Level Findings\n\n"
        f"{result.research_result.regime_summary.conclusion}\n\n"
        f"{table}"
    )


def _build_allocation_profile_section(
    result: MarketResearchWorkflowResult,
) -> str:
    allocation_frame = result.dynamic_target_weight_frame.copy()
    allocation_frame["regime"] = result.regime_labels.astype(int)

    allocation_summary = (
        allocation_frame.groupby("regime")
        .mean(numeric_only=True)
        .sort_index()
        .reset_index()
    )

    headers = ["Regime", *[str(column) for column in allocation_summary.columns[1:]]]
    rows = []

    for row in allocation_summary.to_dict(orient="records"):
        rows.append(
            [
                str(int(row["regime"])),
                *[
                    _format_percent(row[column])
                    for column in allocation_summary.columns[1:]
                ],
            ]
        )

    table = _build_markdown_table(headers=headers, rows=rows)

    return (
        "## Dynamic Allocation Profile\n\n"
        "The dynamic strategy assigns target weights by detected market regime. "
        "Weights are lagged by one period before being applied to returns to "
        "reduce look-ahead bias.\n\n"
        f"{table}"
    )


def _build_research_conclusion_section(
    result: MarketResearchWorkflowResult,
) -> str:
    strategy_summary = result.research_result.strategy_summary
    favorable = strategy_summary.favorable_metric_count
    total = strategy_summary.total_metric_count

    return (
        "## Research Conclusion\n\n"
        f"{strategy_summary.overall_verdict}\n\n"
        f"The dynamic strategy improved {favorable} of {total} assessed "
        "strategy metrics. The regime-level results should be reviewed to "
        "understand whether the improvement is broad-based or concentrated in "
        "specific market environments."
    )


def _build_limitations_section() -> str:
    return (
        "## Limitations\n\n"
        "- Regime labels are model estimates, not directly observable market states.\n"
        "- Historical backtests do not guarantee future performance.\n"
        "- The first workflow uses a simple K-Means regime model and should be "
        "validated before production use.\n"
        "- Transaction costs, liquidity constraints, taxes, and implementation "
        "frictions may reduce realized performance.\n"
        "- Allocation policies should be stress-tested across different market "
        "periods before being used for real capital allocation."
    )


def _build_markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]

    return "\n".join([header_line, separator_line, *row_lines])


def _format_metric_value(metric: str, value: object) -> str:
    clean_metric = metric.strip().lower()

    if clean_metric in PERCENT_METRICS:
        return _format_percent(value)

    return _format_decimal(value)


def _format_relative_delta(value: object) -> str:
    return _format_percent(value)


def _format_percent(value: object) -> str:
    numeric = _to_float(value)

    if pd.isna(numeric):
        return "n/a"

    return f"{numeric * 100:.2f}%"


def _format_decimal(value: object) -> str:
    numeric = _to_float(value)

    if pd.isna(numeric):
        return "n/a"

    return f"{numeric:.4f}"


def _to_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric)
