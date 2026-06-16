from dataclasses import dataclass

import pandas as pd


class InvestmentResearchSummaryError(ValueError):
    """Raised when investment research summaries cannot be created."""


@dataclass(frozen=True, slots=True)
class MetricAssessment:
    """Assessment of one strategy metric."""

    metric: str
    metric_label: str
    benchmark_value: float
    candidate_value: float
    absolute_delta: float
    relative_delta: float | None
    interpretation: str
    is_favorable: bool


@dataclass(frozen=True, slots=True)
class StrategyResearchSummary:
    """Professional summary of candidate strategy performance."""

    benchmark_strategy: str
    candidate_strategy: str
    favorable_metric_count: int
    unfavorable_metric_count: int
    total_metric_count: int
    overall_verdict: str
    metric_assessments: pd.DataFrame


@dataclass(frozen=True, slots=True)
class RegimeResearchSummary:
    """Professional summary of regime-conditioned strategy behavior."""

    regime_summary: pd.DataFrame
    best_regime: int | None
    worst_regime: int | None
    conclusion: str


RETURN_METRICS = {
    "cumulative_return",
    "annualized_return",
    "total_return",
}

RISK_ADJUSTED_METRICS = {
    "sharpe_ratio",
    "sortino_ratio",
}

LOWER_IS_BETTER_METRICS = {
    "annualized_volatility",
    "volatility",
    "max_drawdown",
    "var",
    "cvar",
    "turnover",
    "transaction_cost",
    "transaction_costs",
}

REQUIRED_DELTA_COLUMNS = {
    "metric",
    "metric_label",
    "benchmark_strategy",
    "candidate_strategy",
    "benchmark_value",
    "candidate_value",
    "absolute_delta",
    "relative_delta",
}

REQUIRED_REGIME_COLUMNS = {
    "regime",
    "strategy",
    "metric",
    "value",
}


def build_strategy_research_summary(
    metric_delta_table: pd.DataFrame,
) -> StrategyResearchSummary:
    """Build a professional interpretation of strategy metric deltas."""
    _validate_metric_delta_table(metric_delta_table)

    benchmark_strategy = _single_value(
        metric_delta_table,
        column="benchmark_strategy",
        label="benchmark strategy",
    )
    candidate_strategy = _single_value(
        metric_delta_table,
        column="candidate_strategy",
        label="candidate strategy",
    )

    assessments = [
        _assess_metric_delta(row) for _, row in metric_delta_table.iterrows()
    ]

    assessment_frame = pd.DataFrame(
        [
            {
                "metric": assessment.metric,
                "metric_label": assessment.metric_label,
                "benchmark_value": assessment.benchmark_value,
                "candidate_value": assessment.candidate_value,
                "absolute_delta": assessment.absolute_delta,
                "relative_delta": assessment.relative_delta,
                "interpretation": assessment.interpretation,
                "is_favorable": assessment.is_favorable,
            }
            for assessment in assessments
        ]
    )

    favorable_metric_count = int(assessment_frame["is_favorable"].sum())
    total_metric_count = len(assessment_frame)
    unfavorable_metric_count = total_metric_count - favorable_metric_count

    overall_verdict = _build_overall_verdict(
        candidate_strategy=candidate_strategy,
        benchmark_strategy=benchmark_strategy,
        favorable_metric_count=favorable_metric_count,
        total_metric_count=total_metric_count,
    )

    return StrategyResearchSummary(
        benchmark_strategy=benchmark_strategy,
        candidate_strategy=candidate_strategy,
        favorable_metric_count=favorable_metric_count,
        unfavorable_metric_count=unfavorable_metric_count,
        total_metric_count=total_metric_count,
        overall_verdict=overall_verdict,
        metric_assessments=assessment_frame,
    )


def build_regime_research_summary(
    regime_metric_table: pd.DataFrame,
    candidate_strategy: str,
    benchmark_strategy: str,
    primary_metric: str = "sharpe_ratio",
) -> RegimeResearchSummary:
    """Summarize where the candidate strategy helped or hurt by regime."""
    _validate_regime_metric_table(regime_metric_table)

    clean_candidate = str(candidate_strategy).strip()
    clean_benchmark = str(benchmark_strategy).strip()
    clean_metric = str(primary_metric).strip()

    if not clean_candidate:
        raise InvestmentResearchSummaryError("Candidate strategy must be non-empty")

    if not clean_benchmark:
        raise InvestmentResearchSummaryError("Benchmark strategy must be non-empty")

    if not clean_metric:
        raise InvestmentResearchSummaryError("Primary metric must be non-empty")

    metric_frame = regime_metric_table[
        regime_metric_table["metric"].astype(str) == clean_metric
    ].copy()

    if metric_frame.empty:
        raise InvestmentResearchSummaryError(
            f"Primary regime metric not found: {clean_metric}"
        )

    pivot = metric_frame.pivot_table(
        index="regime",
        columns="strategy",
        values="value",
        aggfunc="first",
    )

    if clean_candidate not in pivot.columns:
        raise InvestmentResearchSummaryError(
            f"Candidate strategy not found in regime table: {clean_candidate}"
        )

    if clean_benchmark not in pivot.columns:
        raise InvestmentResearchSummaryError(
            f"Benchmark strategy not found in regime table: {clean_benchmark}"
        )

    summary = pd.DataFrame(
        {
            "regime": pivot.index.astype(int),
            "benchmark_strategy": clean_benchmark,
            "candidate_strategy": clean_candidate,
            "metric": clean_metric,
            "benchmark_value": pivot[clean_benchmark].to_numpy(dtype=float),
            "candidate_value": pivot[clean_candidate].to_numpy(dtype=float),
        }
    )
    summary["absolute_delta"] = summary["candidate_value"] - summary["benchmark_value"]
    summary["is_favorable"] = summary["absolute_delta"] > 0

    if summary.empty:
        best_regime = None
        worst_regime = None
    else:
        best_regime = int(
            summary.sort_values("absolute_delta", ascending=False).iloc[0]["regime"]
        )
        worst_regime = int(
            summary.sort_values("absolute_delta", ascending=True).iloc[0]["regime"]
        )

    conclusion = _build_regime_conclusion(
        candidate_strategy=clean_candidate,
        benchmark_strategy=clean_benchmark,
        primary_metric=clean_metric,
        summary=summary,
        best_regime=best_regime,
        worst_regime=worst_regime,
    )

    return RegimeResearchSummary(
        regime_summary=summary.reset_index(drop=True),
        best_regime=best_regime,
        worst_regime=worst_regime,
        conclusion=conclusion,
    )


def build_executive_research_summary(
    strategy_summary: StrategyResearchSummary,
    regime_summary: RegimeResearchSummary,
) -> str:
    """Build a concise investment research executive summary."""
    return (
        f"The {strategy_summary.candidate_strategy} strategy was compared against "
        f"the {strategy_summary.benchmark_strategy} benchmark. "
        f"{strategy_summary.overall_verdict} "
        f"{regime_summary.conclusion}"
    )


def _assess_metric_delta(row: pd.Series) -> MetricAssessment:
    metric = str(row["metric"])
    metric_label = str(row["metric_label"])
    benchmark_value = _to_float(row["benchmark_value"], f"{metric} benchmark value")
    candidate_value = _to_float(row["candidate_value"], f"{metric} candidate value")
    absolute_delta = _to_float(row["absolute_delta"], f"{metric} absolute delta")
    relative_delta = _to_optional_float(row["relative_delta"])

    is_favorable = _is_metric_delta_favorable(
        metric=metric,
        absolute_delta=absolute_delta,
    )

    interpretation = _interpret_metric_delta(
        metric_label=metric_label,
        absolute_delta=absolute_delta,
        is_favorable=is_favorable,
    )

    return MetricAssessment(
        metric=metric,
        metric_label=metric_label,
        benchmark_value=benchmark_value,
        candidate_value=candidate_value,
        absolute_delta=absolute_delta,
        relative_delta=relative_delta,
        interpretation=interpretation,
        is_favorable=is_favorable,
    )


def _is_metric_delta_favorable(metric: str, absolute_delta: float) -> bool:
    clean_metric = metric.strip().lower()

    if clean_metric in LOWER_IS_BETTER_METRICS:
        return absolute_delta < 0

    if clean_metric in RETURN_METRICS:
        return absolute_delta > 0

    if clean_metric in RISK_ADJUSTED_METRICS:
        return absolute_delta > 0

    return absolute_delta > 0


def _interpret_metric_delta(
    metric_label: str,
    absolute_delta: float,
    is_favorable: bool,
) -> str:
    direction = "improved" if is_favorable else "deteriorated"

    if absolute_delta == 0:
        return f"{metric_label} was unchanged versus the benchmark."

    return f"{metric_label} {direction} versus the benchmark by {absolute_delta:.4f}."


def _build_overall_verdict(
    candidate_strategy: str,
    benchmark_strategy: str,
    favorable_metric_count: int,
    total_metric_count: int,
) -> str:
    favorable_share = favorable_metric_count / total_metric_count

    if favorable_share >= 0.75:
        return (
            f"Overall, {candidate_strategy} showed broad improvement over "
            f"{benchmark_strategy}, with {favorable_metric_count} of "
            f"{total_metric_count} assessed metrics moving in a favorable direction."
        )

    if favorable_share >= 0.50:
        return (
            f"Overall, {candidate_strategy} showed mixed but positive evidence "
            f"relative to {benchmark_strategy}, with {favorable_metric_count} of "
            f"{total_metric_count} assessed metrics moving in a favorable direction."
        )

    return (
        f"Overall, {candidate_strategy} did not provide enough evidence of "
        f"improvement over {benchmark_strategy}, with only "
        f"{favorable_metric_count} of {total_metric_count} assessed metrics "
        f"moving in a favorable direction."
    )


def _build_regime_conclusion(
    candidate_strategy: str,
    benchmark_strategy: str,
    primary_metric: str,
    summary: pd.DataFrame,
    best_regime: int | None,
    worst_regime: int | None,
) -> str:
    favorable_count = int(summary["is_favorable"].sum())
    total_count = len(summary)

    if total_count == 0:
        return "No regime-level conclusion could be formed."

    conclusion = (
        f"Regime-level analysis using {primary_metric} showed that "
        f"{candidate_strategy} outperformed {benchmark_strategy} in "
        f"{favorable_count} of {total_count} regimes."
    )

    if best_regime is not None:
        conclusion += f" The strongest relative regime was regime {best_regime}."

    if worst_regime is not None:
        conclusion += f" The weakest relative regime was regime {worst_regime}."

    return conclusion


def _validate_metric_delta_table(metric_delta_table: pd.DataFrame) -> None:
    if metric_delta_table.empty:
        raise InvestmentResearchSummaryError("Metric delta table cannot be empty")

    missing_columns = REQUIRED_DELTA_COLUMNS.difference(metric_delta_table.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise InvestmentResearchSummaryError(
            f"Missing metric delta column(s): {missing}"
        )

    if metric_delta_table["metric"].isna().any():
        raise InvestmentResearchSummaryError("Metric delta table has missing metrics")


def _validate_regime_metric_table(regime_metric_table: pd.DataFrame) -> None:
    if regime_metric_table.empty:
        raise InvestmentResearchSummaryError("Regime metric table cannot be empty")

    missing_columns = REQUIRED_REGIME_COLUMNS.difference(regime_metric_table.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise InvestmentResearchSummaryError(
            f"Missing regime metric column(s): {missing}"
        )


def _single_value(data: pd.DataFrame, column: str, label: str) -> str:
    values = data[column].dropna().astype(str).unique()

    if len(values) != 1:
        raise InvestmentResearchSummaryError(
            f"Expected exactly one {label}, found {len(values)}"
        )

    return str(values[0])


def _to_float(value: object, context: str) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise InvestmentResearchSummaryError(f"Non-numeric value for {context}")

    return float(numeric)


def _to_optional_float(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        return None

    return float(numeric)
