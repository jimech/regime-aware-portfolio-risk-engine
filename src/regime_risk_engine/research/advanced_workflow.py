from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.research.advanced_memo import (
    AdvancedResearchMemoConfig,
    AdvancedResearchMemoInputs,
    build_advanced_research_memo,
)
from regime_risk_engine.research.attribution import (
    StrategyAttributionSummary,
    build_strategy_attribution_from_workflow,
)
from regime_risk_engine.research.factor_exposure import (
    FactorExposureSummary,
    build_factor_exposure_summary,
)
from regime_risk_engine.research.market_workflow import MarketResearchWorkflowResult
from regime_risk_engine.research.memo import (
    MarketResearchMemoConfig,
    build_market_research_memo,
)
from regime_risk_engine.research.regime_intelligence import (
    RegimeIntelligenceSummary,
    build_regime_intelligence_from_workflow,
)
from regime_risk_engine.research.regime_transitions import (
    RegimeTransitionSummary,
    build_regime_transition_summary,
)
from regime_risk_engine.research.rolling_factor_exposure import (
    RollingFactorExposureResult,
    estimate_rolling_factor_exposures,
)
from regime_risk_engine.research.scenario_simulation import (
    RegimeScenarioSimulationConfig,
    RegimeScenarioSimulationResult,
    simulate_regime_strategy_scenarios,
)
from regime_risk_engine.research.stress_testing import (
    StressPeriod,
    StressTestSummary,
    build_stress_test_summary,
)


class AdvancedResearchWorkflowError(ValueError):
    """Raised when the advanced research workflow cannot be built."""


@dataclass(frozen=True, slots=True)
class AdvancedResearchWorkflowResult:
    """Advanced research workflow outputs."""

    base_memo: str
    regime_intelligence: RegimeIntelligenceSummary
    regime_transitions: RegimeTransitionSummary
    stress_test: StressTestSummary | None
    attribution: StrategyAttributionSummary
    factor_exposure: FactorExposureSummary | None
    rolling_factor_exposure: RollingFactorExposureResult | None
    scenario_simulation: RegimeScenarioSimulationResult
    advanced_inputs: AdvancedResearchMemoInputs
    advanced_memo: str


def build_advanced_research_workflow(
    market_result: MarketResearchWorkflowResult,
    stress_periods: Sequence[StressPeriod] | None = None,
    factor_returns: pd.DataFrame | None = None,
    market_memo_config: MarketResearchMemoConfig | None = None,
    advanced_memo_config: AdvancedResearchMemoConfig | None = None,
    scenario_config: RegimeScenarioSimulationConfig | None = None,
    annualization_factor: int = 252,
    rolling_factor_window: int = 20,
) -> AdvancedResearchWorkflowResult:
    """Build advanced research outputs from a market workflow result."""
    _validate_market_result(market_result)

    strategy_regime_return_frame = _build_strategy_regime_return_frame(market_result)

    base_memo = build_market_research_memo(
        result=market_result,
        config=market_memo_config,
    )
    regime_intelligence = build_regime_intelligence_from_workflow(
        result=market_result,
        annualization_factor=annualization_factor,
    )
    regime_transitions = build_regime_transition_summary(
        regime_labels=market_result.regime_labels,
    )
    attribution = build_strategy_attribution_from_workflow(market_result)
    scenario_simulation = simulate_regime_strategy_scenarios(
        return_frame=strategy_regime_return_frame,
        config=scenario_config,
    )

    stress_test = None

    if stress_periods is not None:
        stress_test = build_stress_test_summary(
            return_frame=strategy_regime_return_frame,
            stress_periods=list(stress_periods),
            benchmark_strategy="static",
            candidate_strategy="dynamic",
            annualization_factor=annualization_factor,
        )

    factor_exposure = None
    rolling_factor_exposure = None

    if factor_returns is not None:
        factor_exposure = build_factor_exposure_summary(
            strategy_returns=strategy_regime_return_frame[["static", "dynamic"]],
            factor_returns=factor_returns,
            regime_labels=strategy_regime_return_frame["regime"],
            annualization_factor=annualization_factor,
        )
        rolling_factor_exposure = estimate_rolling_factor_exposures(
            strategy_returns=_build_dynamic_strategy_return_input(
                strategy_regime_return_frame
            ),
            factor_returns=_build_factor_return_input(factor_returns),
            window=rolling_factor_window,
        )

    advanced_inputs = AdvancedResearchMemoInputs(
        base_memo=base_memo,
        regime_intelligence=regime_intelligence,
        regime_transitions=regime_transitions,
        stress_test=stress_test,
        attribution=attribution,
        factor_exposure=factor_exposure,
        rolling_factor_exposure=rolling_factor_exposure,
        scenario_simulation=scenario_simulation,
    )
    advanced_memo = build_advanced_research_memo(
        inputs=advanced_inputs,
        config=advanced_memo_config,
    )

    return AdvancedResearchWorkflowResult(
        base_memo=base_memo,
        regime_intelligence=regime_intelligence,
        regime_transitions=regime_transitions,
        stress_test=stress_test,
        attribution=attribution,
        factor_exposure=factor_exposure,
        rolling_factor_exposure=rolling_factor_exposure,
        scenario_simulation=scenario_simulation,
        advanced_inputs=advanced_inputs,
        advanced_memo=advanced_memo,
    )


def _validate_market_result(market_result: MarketResearchWorkflowResult) -> None:
    if market_result.asset_returns.empty:
        raise AdvancedResearchWorkflowError("Asset returns cannot be empty")

    if market_result.regime_labels.empty:
        raise AdvancedResearchWorkflowError("Regime labels cannot be empty")

    if market_result.strategy_comparison.return_comparison.empty:
        raise AdvancedResearchWorkflowError(
            "Strategy return comparison cannot be empty"
        )

    required_columns = {"static", "dynamic"}
    missing_columns = required_columns.difference(
        market_result.strategy_comparison.return_comparison.columns
    )

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise AdvancedResearchWorkflowError(
            f"Missing strategy return column(s): {missing}"
        )


def _build_strategy_regime_return_frame(
    market_result: MarketResearchWorkflowResult,
) -> pd.DataFrame:
    return_comparison = market_result.strategy_comparison.return_comparison
    regime_labels = market_result.regime_labels

    common_index = return_comparison.index.intersection(regime_labels.index)

    if common_index.empty:
        raise AdvancedResearchWorkflowError(
            "Strategy returns and regime labels have no overlapping dates"
        )

    frame = return_comparison.loc[common_index, ["static", "dynamic"]].copy()
    frame["regime"] = regime_labels.loc[common_index].astype(int)

    return frame


def _build_dynamic_strategy_return_input(
    strategy_regime_return_frame: pd.DataFrame,
) -> pd.DataFrame:
    strategy_returns = strategy_regime_return_frame[["dynamic"]].rename(
        columns={"dynamic": "return"}
    )
    strategy_returns.index.name = "date"

    return strategy_returns.reset_index()


def _build_factor_return_input(factor_returns: pd.DataFrame) -> pd.DataFrame:
    factor_return_input = factor_returns.copy()
    factor_return_input.index.name = "date"

    return factor_return_input.reset_index()
