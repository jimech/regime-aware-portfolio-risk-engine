from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

from regime_risk_engine.research.market_workflow import MarketResearchWorkflowResult


class RegimeIntelligenceError(ValueError):
    """Raised when regime intelligence cannot be created."""


@dataclass(frozen=True, slots=True)
class RegimeIntelligenceSummary:
    """Interpretable financial summary of detected market regimes."""

    profile_table: pd.DataFrame
    strongest_regime: int | None
    weakest_regime: int | None
    stress_regime: int | None
    narrative: str


DEFAULT_ASSET_ROLES = {
    "SPY": "equity",
    "VTI": "equity",
    "QQQ": "equity",
    "IWM": "equity",
    "EFA": "equity",
    "EEM": "equity",
    "VNQ": "real_asset",
    "TLT": "defensive",
    "IEF": "defensive",
    "SHY": "defensive",
    "BND": "defensive",
    "AGG": "defensive",
    "GLD": "real_asset",
    "SLV": "real_asset",
    "DBC": "real_asset",
    "USO": "real_asset",
    "HYG": "credit",
    "LQD": "credit",
}

EQUITY_ROLES = {"equity"}
DEFENSIVE_ROLES = {"defensive"}
REAL_ASSET_ROLES = {"real_asset"}


def build_regime_intelligence_from_workflow(
    result: MarketResearchWorkflowResult,
    asset_roles: Mapping[str, str] | None = None,
    annualization_factor: int = 252,
) -> RegimeIntelligenceSummary:
    """Build regime intelligence from a market research workflow result."""
    return build_regime_intelligence_summary(
        asset_returns=result.asset_returns,
        regime_labels=result.regime_labels,
        asset_roles=asset_roles,
        annualization_factor=annualization_factor,
    )


def build_regime_intelligence_summary(
    asset_returns: pd.DataFrame,
    regime_labels: pd.Series,
    asset_roles: Mapping[str, str] | None = None,
    annualization_factor: int = 252,
) -> RegimeIntelligenceSummary:
    """Build financial labels and interpretations for detected regimes."""
    _validate_inputs(
        asset_returns=asset_returns,
        regime_labels=regime_labels,
        annualization_factor=annualization_factor,
    )

    aligned_returns, aligned_regimes = _align_returns_and_regimes(
        asset_returns=asset_returns,
        regime_labels=regime_labels,
    )
    clean_asset_roles = _resolve_asset_roles(
        tickers=[str(column) for column in aligned_returns.columns],
        asset_roles=asset_roles,
    )

    profile_rows = []

    for regime, regime_returns in aligned_returns.groupby(aligned_regimes, sort=True):
        profile_rows.append(
            _build_regime_profile_row(
                regime=int(regime),
                regime_returns=regime_returns,
                asset_roles=clean_asset_roles,
                annualization_factor=annualization_factor,
            )
        )

    profile_table = (
        pd.DataFrame(profile_rows).sort_values("regime").reset_index(drop=True)
    )

    median_volatility = float(profile_table["annualized_volatility"].median())

    profile_table["label"] = [
        _classify_regime(row=row, median_volatility=median_volatility)
        for row in profile_table.to_dict(orient="records")
    ]
    profile_table["interpretation"] = [
        _interpret_regime(row=row) for row in profile_table.to_dict(orient="records")
    ]

    strongest_regime = _extract_regime_by_sort(
        profile_table=profile_table,
        sort_column="annualized_return",
        ascending=False,
    )
    weakest_regime = _extract_regime_by_sort(
        profile_table=profile_table,
        sort_column="annualized_return",
        ascending=True,
    )
    stress_regime = _extract_regime_by_sort(
        profile_table=profile_table,
        sort_column="max_drawdown",
        ascending=True,
    )

    narrative = _build_regime_narrative(
        profile_table=profile_table,
        strongest_regime=strongest_regime,
        weakest_regime=weakest_regime,
        stress_regime=stress_regime,
    )

    return RegimeIntelligenceSummary(
        profile_table=profile_table,
        strongest_regime=strongest_regime,
        weakest_regime=weakest_regime,
        stress_regime=stress_regime,
        narrative=narrative,
    )


def _validate_inputs(
    asset_returns: pd.DataFrame,
    regime_labels: pd.Series,
    annualization_factor: int,
) -> None:
    if asset_returns.empty:
        raise RegimeIntelligenceError("Asset returns cannot be empty")

    if regime_labels.empty:
        raise RegimeIntelligenceError("Regime labels cannot be empty")

    if annualization_factor <= 0:
        raise RegimeIntelligenceError("Annualization factor must be positive")

    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        raise RegimeIntelligenceError("Asset returns index must be a DatetimeIndex")

    if not isinstance(regime_labels.index, pd.DatetimeIndex):
        raise RegimeIntelligenceError("Regime labels index must be a DatetimeIndex")

    numeric_returns = asset_returns.apply(pd.to_numeric, errors="coerce")

    if numeric_returns.isna().any().any():
        raise RegimeIntelligenceError("Asset returns must be numeric and complete")

    if regime_labels.isna().any():
        raise RegimeIntelligenceError("Regime labels cannot contain missing values")


def _align_returns_and_regimes(
    asset_returns: pd.DataFrame,
    regime_labels: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    common_index = asset_returns.index.intersection(regime_labels.index)

    if common_index.empty:
        raise RegimeIntelligenceError(
            "Asset returns and regime labels have no overlapping dates"
        )

    aligned_returns = asset_returns.loc[common_index].copy()
    aligned_regimes = regime_labels.loc[common_index].astype(int).copy()

    if aligned_returns.empty:
        raise RegimeIntelligenceError("No aligned return observations are available")

    if aligned_regimes.nunique() < 2:
        raise RegimeIntelligenceError("At least two regimes are required")

    return aligned_returns, aligned_regimes


def _resolve_asset_roles(
    tickers: list[str],
    asset_roles: Mapping[str, str] | None,
) -> dict[str, str]:
    clean_roles = {
        ticker.upper(): DEFAULT_ASSET_ROLES.get(ticker.upper(), "other")
        for ticker in tickers
    }

    if asset_roles is None:
        return clean_roles

    override_roles = {
        str(ticker).strip().upper(): str(role).strip().lower()
        for ticker, role in asset_roles.items()
    }

    unknown_tickers = set(override_roles).difference(clean_roles)

    if unknown_tickers:
        raise RegimeIntelligenceError(
            f"Asset role overrides contain unknown ticker(s): {sorted(unknown_tickers)}"
        )

    empty_roles = [
        ticker for ticker, role in override_roles.items() if not role.strip()
    ]

    if empty_roles:
        raise RegimeIntelligenceError(
            f"Asset role overrides contain empty role(s): {sorted(empty_roles)}"
        )

    clean_roles.update(override_roles)

    return clean_roles


def _build_regime_profile_row(
    regime: int,
    regime_returns: pd.DataFrame,
    asset_roles: Mapping[str, str],
    annualization_factor: int,
) -> dict[str, object]:
    equal_weight_returns = regime_returns.mean(axis=1)
    cumulative_asset_returns = (1.0 + regime_returns).prod() - 1.0

    cumulative_return = float((1.0 + equal_weight_returns).prod() - 1.0)
    annualized_return = float(
        (1.0 + cumulative_return) ** (annualization_factor / len(equal_weight_returns))
        - 1.0
    )

    if len(equal_weight_returns) < 2:
        annualized_volatility = 0.0
    else:
        annualized_volatility = float(
            equal_weight_returns.std(ddof=1) * sqrt(annualization_factor)
        )

    if annualized_volatility == 0.0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float(annualized_return / annualized_volatility)

    wealth = (1.0 + equal_weight_returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    average_correlation = _calculate_average_correlation(regime_returns)

    best_asset = str(cumulative_asset_returns.idxmax())
    worst_asset = str(cumulative_asset_returns.idxmin())

    return {
        "regime": regime,
        "observation_count": int(len(regime_returns)),
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "average_correlation": average_correlation,
        "equity_return": _calculate_role_return(
            regime_returns=regime_returns,
            asset_roles=asset_roles,
            target_roles=EQUITY_ROLES,
        ),
        "defensive_return": _calculate_role_return(
            regime_returns=regime_returns,
            asset_roles=asset_roles,
            target_roles=DEFENSIVE_ROLES,
        ),
        "real_asset_return": _calculate_role_return(
            regime_returns=regime_returns,
            asset_roles=asset_roles,
            target_roles=REAL_ASSET_ROLES,
        ),
        "best_asset": best_asset,
        "worst_asset": worst_asset,
    }


def _calculate_average_correlation(regime_returns: pd.DataFrame) -> float:
    if len(regime_returns.columns) < 2:
        return 0.0

    correlation_matrix = regime_returns.corr().to_numpy(dtype=float)
    mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
    values = correlation_matrix[mask]

    average_correlation = float(np.nanmean(values))

    if np.isnan(average_correlation):
        return 0.0

    return average_correlation


def _calculate_role_return(
    regime_returns: pd.DataFrame,
    asset_roles: Mapping[str, str],
    target_roles: set[str],
) -> float:
    role_columns = [
        column
        for column in regime_returns.columns
        if asset_roles[str(column).upper()] in target_roles
    ]

    if not role_columns:
        return 0.0

    role_returns = regime_returns[role_columns].mean(axis=1)

    return float(role_returns.mean())


def _classify_regime(row: dict[str, object], median_volatility: float) -> str:
    annualized_return = _to_float(row["annualized_return"])
    annualized_volatility = _to_float(row["annualized_volatility"])
    max_drawdown = _to_float(row["max_drawdown"])
    equity_return = _to_float(row["equity_return"])
    defensive_return = _to_float(row["defensive_return"])
    real_asset_return = _to_float(row["real_asset_return"])

    if annualized_return < 0.0 or max_drawdown <= -0.02:
        return "Defensive / stress"

    if (
        real_asset_return > equity_return
        and real_asset_return > defensive_return
        and real_asset_return > 0.0
    ):
        return "Inflation / real assets"

    if (
        equity_return > defensive_return
        and equity_return > real_asset_return
        and annualized_return > 0.0
    ):
        return "Growth / risk-on"

    if annualized_volatility <= median_volatility and annualized_return >= 0.0:
        return "Low-volatility grind"

    return "Mixed / transition"


def _interpret_regime(row: dict[str, object]) -> str:
    label = str(row["label"])
    best_asset = str(row["best_asset"])
    worst_asset = str(row["worst_asset"])

    if label == "Growth / risk-on":
        return (
            f"Equity leadership dominated this regime. {best_asset} was the "
            f"strongest asset, while {worst_asset} lagged."
        )

    if label == "Defensive / stress":
        return (
            f"This regime showed defensive or stress characteristics. "
            f"{best_asset} held up best, while {worst_asset} was the weakest asset."
        )

    if label == "Inflation / real assets":
        return (
            f"Real assets led this regime. {best_asset} was the strongest asset, "
            f"while {worst_asset} lagged."
        )

    if label == "Low-volatility grind":
        return (
            f"This regime showed relatively calm market behavior. {best_asset} "
            f"was the strongest asset, while {worst_asset} lagged."
        )

    return (
        f"This regime showed mixed market behavior. {best_asset} was the strongest "
        f"asset, while {worst_asset} lagged."
    )


def _extract_regime_by_sort(
    profile_table: pd.DataFrame,
    sort_column: str,
    ascending: bool,
) -> int | None:
    if profile_table.empty:
        return None

    sorted_table = profile_table.sort_values(sort_column, ascending=ascending)

    return int(sorted_table.iloc[0]["regime"])


def _build_regime_narrative(
    profile_table: pd.DataFrame,
    strongest_regime: int | None,
    weakest_regime: int | None,
    stress_regime: int | None,
) -> str:
    label_map = {
        int(row["regime"]): str(row["label"])
        for row in profile_table.to_dict(orient="records")
    }

    parts = [
        "The detected regimes were converted into interpretable market states "
        "using returns, volatility, drawdown, correlation, and asset leadership."
    ]

    if strongest_regime is not None:
        parts.append(
            f"The strongest regime was regime {strongest_regime}, classified as "
            f"{label_map[strongest_regime]}."
        )

    if weakest_regime is not None:
        parts.append(
            f"The weakest return regime was regime {weakest_regime}, classified as "
            f"{label_map[weakest_regime]}."
        )

    if stress_regime is not None:
        parts.append(
            f"The most severe drawdown regime was regime {stress_regime}, "
            f"classified as {label_map[stress_regime]}."
        )

    return " ".join(parts)


def _to_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise RegimeIntelligenceError("Expected numeric regime profile value")

    return float(numeric)
