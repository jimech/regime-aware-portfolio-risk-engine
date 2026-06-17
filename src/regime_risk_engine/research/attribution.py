from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.research.market_workflow import MarketResearchWorkflowResult


class StrategyAttributionError(ValueError):
    """Raised when strategy attribution cannot be completed."""


@dataclass(frozen=True, slots=True)
class StrategyAttributionSummary:
    """Strategy attribution result."""

    asset_attribution: pd.DataFrame
    regime_attribution: pd.DataFrame | None
    top_positive_asset: str | None
    top_negative_asset: str | None
    strongest_regime: int | None
    weakest_regime: int | None
    narrative: str


def build_strategy_attribution_from_workflow(
    result: MarketResearchWorkflowResult,
) -> StrategyAttributionSummary:
    """Build strategy attribution from a market research workflow result."""
    return build_strategy_attribution_summary(
        asset_returns=result.asset_returns,
        static_weight_frame=result.static_weight_frame,
        dynamic_weight_frame=result.dynamic_applied_weight_frame,
        regime_labels=result.regime_labels,
    )


def build_strategy_attribution_summary(
    asset_returns: pd.DataFrame,
    static_weight_frame: pd.DataFrame,
    dynamic_weight_frame: pd.DataFrame,
    regime_labels: pd.Series | None = None,
) -> StrategyAttributionSummary:
    """Explain dynamic strategy excess return by asset and regime."""
    _validate_inputs(
        asset_returns=asset_returns,
        static_weight_frame=static_weight_frame,
        dynamic_weight_frame=dynamic_weight_frame,
        regime_labels=regime_labels,
    )

    aligned_returns, aligned_static_weights, aligned_dynamic_weights = (
        _align_return_and_weight_frames(
            asset_returns=asset_returns,
            static_weight_frame=static_weight_frame,
            dynamic_weight_frame=dynamic_weight_frame,
        )
    )

    aligned_regimes = None

    if regime_labels is not None:
        aligned_regimes = _align_regime_labels(
            regime_labels=regime_labels,
            index=aligned_returns.index,
        )

    active_weight_frame = aligned_dynamic_weights - aligned_static_weights
    static_contribution_frame = aligned_static_weights * aligned_returns
    dynamic_contribution_frame = aligned_dynamic_weights * aligned_returns
    active_contribution_frame = active_weight_frame * aligned_returns

    asset_attribution = _build_asset_attribution_table(
        static_weight_frame=aligned_static_weights,
        dynamic_weight_frame=aligned_dynamic_weights,
        static_contribution_frame=static_contribution_frame,
        dynamic_contribution_frame=dynamic_contribution_frame,
        active_contribution_frame=active_contribution_frame,
    )

    regime_attribution = None

    if aligned_regimes is not None:
        regime_attribution = _build_regime_attribution_table(
            static_contribution_frame=static_contribution_frame,
            dynamic_contribution_frame=dynamic_contribution_frame,
            active_contribution_frame=active_contribution_frame,
            active_weight_frame=active_weight_frame,
            regime_labels=aligned_regimes,
        )

    top_positive_asset = _extract_top_positive_asset(asset_attribution)
    top_negative_asset = _extract_top_negative_asset(asset_attribution)
    strongest_regime = _extract_extreme_regime(
        regime_attribution=regime_attribution,
        ascending=False,
    )
    weakest_regime = _extract_extreme_regime(
        regime_attribution=regime_attribution,
        ascending=True,
    )

    narrative = _build_narrative(
        asset_attribution=asset_attribution,
        regime_attribution=regime_attribution,
        top_positive_asset=top_positive_asset,
        top_negative_asset=top_negative_asset,
        strongest_regime=strongest_regime,
        weakest_regime=weakest_regime,
    )

    return StrategyAttributionSummary(
        asset_attribution=asset_attribution,
        regime_attribution=regime_attribution,
        top_positive_asset=top_positive_asset,
        top_negative_asset=top_negative_asset,
        strongest_regime=strongest_regime,
        weakest_regime=weakest_regime,
        narrative=narrative,
    )


def _validate_inputs(
    asset_returns: pd.DataFrame,
    static_weight_frame: pd.DataFrame,
    dynamic_weight_frame: pd.DataFrame,
    regime_labels: pd.Series | None,
) -> None:
    if asset_returns.empty:
        raise StrategyAttributionError("Asset returns cannot be empty")

    if static_weight_frame.empty:
        raise StrategyAttributionError("Static weight frame cannot be empty")

    if dynamic_weight_frame.empty:
        raise StrategyAttributionError("Dynamic weight frame cannot be empty")

    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise StrategyAttributionError("Asset returns index must be a DatetimeIndex")

    if not isinstance(static_weight_frame.index, pd.DatetimeIndex):
        raise StrategyAttributionError(
            "Static weight frame index must be a DatetimeIndex"
        )

    if not isinstance(dynamic_weight_frame.index, pd.DatetimeIndex):
        raise StrategyAttributionError(
            "Dynamic weight frame index must be a DatetimeIndex"
        )

    if regime_labels is not None and not isinstance(
        regime_labels.index,
        pd.DatetimeIndex,
    ):
        raise StrategyAttributionError("Regime labels index must be a DatetimeIndex")

    expected_assets = set(asset_returns.columns)

    if set(static_weight_frame.columns) != expected_assets:
        raise StrategyAttributionError(
            "Static weight frame columns must match asset return columns"
        )

    if set(dynamic_weight_frame.columns) != expected_assets:
        raise StrategyAttributionError(
            "Dynamic weight frame columns must match asset return columns"
        )

    _validate_numeric_frame(asset_returns, "Asset returns")
    _validate_numeric_frame(static_weight_frame, "Static weight frame")
    _validate_numeric_frame(dynamic_weight_frame, "Dynamic weight frame")

    if regime_labels is not None and regime_labels.isna().any():
        raise StrategyAttributionError("Regime labels cannot contain missing values")


def _validate_numeric_frame(frame: pd.DataFrame, label: str) -> None:
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")

    if numeric_frame.isna().any().any():
        raise StrategyAttributionError(f"{label} must be numeric and complete")


def _align_return_and_weight_frames(
    asset_returns: pd.DataFrame,
    static_weight_frame: pd.DataFrame,
    dynamic_weight_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_index = asset_returns.index.intersection(static_weight_frame.index)
    common_index = common_index.intersection(dynamic_weight_frame.index)

    if common_index.empty:
        raise StrategyAttributionError(
            "Asset returns and weight frames have no overlapping dates"
        )

    columns = [str(column) for column in asset_returns.columns]

    aligned_returns = asset_returns.loc[common_index, columns].astype(float).copy()
    aligned_static_weights = (
        static_weight_frame.loc[common_index, columns].astype(float).copy()
    )
    aligned_dynamic_weights = (
        dynamic_weight_frame.loc[common_index, columns].astype(float).copy()
    )

    return aligned_returns, aligned_static_weights, aligned_dynamic_weights


def _align_regime_labels(
    regime_labels: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    common_index = index.intersection(regime_labels.index)

    if common_index.empty:
        raise StrategyAttributionError(
            "Regime labels have no overlapping dates with attribution data"
        )

    return regime_labels.loc[index].astype(int).copy()


def _build_asset_attribution_table(
    static_weight_frame: pd.DataFrame,
    dynamic_weight_frame: pd.DataFrame,
    static_contribution_frame: pd.DataFrame,
    dynamic_contribution_frame: pd.DataFrame,
    active_contribution_frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for asset in static_weight_frame.columns:
        rows.append(
            {
                "asset": str(asset),
                "average_static_weight": float(static_weight_frame[asset].mean()),
                "average_dynamic_weight": float(dynamic_weight_frame[asset].mean()),
                "average_active_weight": float(
                    dynamic_weight_frame[asset].mean()
                    - static_weight_frame[asset].mean()
                ),
                "static_return_contribution": float(
                    static_contribution_frame[asset].sum()
                ),
                "dynamic_return_contribution": float(
                    dynamic_contribution_frame[asset].sum()
                ),
                "active_return_contribution": float(
                    active_contribution_frame[asset].sum()
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("active_return_contribution", ascending=False)
        .reset_index(drop=True)
    )


def _build_regime_attribution_table(
    static_contribution_frame: pd.DataFrame,
    dynamic_contribution_frame: pd.DataFrame,
    active_contribution_frame: pd.DataFrame,
    active_weight_frame: pd.DataFrame,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    rows = []

    for regime, regime_index in regime_labels.groupby(regime_labels).groups.items():
        regime_dates = pd.DatetimeIndex(regime_index)

        static_returns = static_contribution_frame.loc[regime_dates].sum(axis=1)
        dynamic_returns = dynamic_contribution_frame.loc[regime_dates].sum(axis=1)
        active_contributions = active_contribution_frame.loc[regime_dates]
        active_weights = active_weight_frame.loc[regime_dates]

        asset_contributions = active_contributions.sum()
        average_active_weights = active_weights.mean()

        rows.append(
            {
                "regime": int(regime),
                "observation_count": int(len(regime_dates)),
                "static_cumulative_return": float((1.0 + static_returns).prod() - 1.0),
                "dynamic_cumulative_return": float(
                    (1.0 + dynamic_returns).prod() - 1.0
                ),
                "active_return_contribution": float(asset_contributions.sum()),
                "top_positive_asset": _extract_top_asset(asset_contributions),
                "top_negative_asset": _extract_bottom_asset(asset_contributions),
                "largest_overweight_asset": _extract_top_asset(average_active_weights),
                "largest_underweight_asset": _extract_bottom_asset(
                    average_active_weights
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def _extract_top_asset(values: pd.Series) -> str | None:
    if values.empty:
        return None

    return str(values.idxmax())


def _extract_bottom_asset(values: pd.Series) -> str | None:
    if values.empty:
        return None

    return str(values.idxmin())


def _extract_top_positive_asset(asset_attribution: pd.DataFrame) -> str | None:
    if asset_attribution.empty:
        return None

    top_row = asset_attribution.sort_values(
        "active_return_contribution",
        ascending=False,
    ).iloc[0]

    top_contribution = _to_float(top_row["active_return_contribution"])

    if top_contribution <= 0.0:
        return None

    return str(top_row["asset"])


def _extract_top_negative_asset(asset_attribution: pd.DataFrame) -> str | None:
    if asset_attribution.empty:
        return None

    bottom_row = asset_attribution.sort_values(
        "active_return_contribution",
        ascending=True,
    ).iloc[0]

    bottom_contribution = _to_float(bottom_row["active_return_contribution"])

    if bottom_contribution >= 0.0:
        return None

    return str(bottom_row["asset"])


def _extract_extreme_regime(
    regime_attribution: pd.DataFrame | None,
    ascending: bool,
) -> int | None:
    if regime_attribution is None or regime_attribution.empty:
        return None

    sorted_table = regime_attribution.sort_values(
        "active_return_contribution",
        ascending=ascending,
    )

    return int(sorted_table.iloc[0]["regime"])


def _build_narrative(
    asset_attribution: pd.DataFrame,
    regime_attribution: pd.DataFrame | None,
    top_positive_asset: str | None,
    top_negative_asset: str | None,
    strongest_regime: int | None,
    weakest_regime: int | None,
) -> str:
    total_active_contribution = float(
        asset_attribution["active_return_contribution"].sum()
    )

    narrative = (
        "Strategy attribution decomposed dynamic-versus-static performance into "
        f"asset-level active return contributions. Total active contribution was "
        f"{total_active_contribution:.4f}."
    )

    if top_positive_asset is not None:
        narrative += (
            f" The strongest positive asset contributor was {top_positive_asset}."
        )

    if top_negative_asset is not None:
        narrative += (
            f" The largest negative asset contributor was {top_negative_asset}."
        )

    if regime_attribution is not None and strongest_regime is not None:
        narrative += f" Regime attribution was strongest in regime {strongest_regime}."

    if regime_attribution is not None and weakest_regime is not None:
        narrative += f" Regime attribution was weakest in regime {weakest_regime}."

    return narrative


def _to_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise StrategyAttributionError("Expected numeric attribution value")

    return float(numeric)
