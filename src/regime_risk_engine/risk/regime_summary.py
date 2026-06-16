from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from regime_risk_engine.risk.metrics import (
    DEFAULT_ANNUALIZATION_FACTOR,
    DEFAULT_CONFIDENCE_LEVEL,
    summarize_risk_metrics,
)


class RegimeRiskSummaryError(ValueError):
    """Raised when regime risk summaries cannot be calculated."""


@dataclass(frozen=True, slots=True)
class RegimeRiskSummary:
    """Container for regime-level risk summaries."""

    portfolio_summary: pd.DataFrame
    asset_summary: pd.DataFrame


def calculate_portfolio_returns(
    returns: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    return_col: str = "return",
) -> pd.Series:
    """Calculate portfolio returns from long-format asset returns.

    Args:
        returns: Long-format returns with date, ticker, and return columns.
        weights: Optional ticker weights. If omitted, equal weights are used.
        return_col: Name of the return column.

    Returns:
        Date-indexed portfolio return series.
    """
    _validate_returns(returns, return_col=return_col)

    wide_returns = _pivot_returns(returns, return_col=return_col)

    if weights is None:
        portfolio_returns = wide_returns.mean(axis=1)
    else:
        clean_weights = _validate_weights(weights, list(wide_returns.columns))
        weight_series = pd.Series(clean_weights, dtype=float)
        portfolio_returns = wide_returns.mul(weight_series, axis=1).sum(axis=1)

    portfolio_returns.name = "portfolio_return"

    return portfolio_returns.sort_index()


def calculate_regime_portfolio_risk_summary(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    weights: Mapping[str, float] | None = None,
    risk_free_rate: float = 0.0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate portfolio risk metrics grouped by regime."""
    portfolio_returns = calculate_portfolio_returns(
        returns=returns,
        weights=weights,
        return_col=return_col,
    )
    labels = _prepare_regime_labels(regime_labels)

    portfolio_returns, labels = _align_returns_and_labels(portfolio_returns, labels)

    rows: list[dict[str, object]] = []

    for regime in sorted(labels.unique()):
        regime_dates = labels[labels == regime].index
        regime_returns = portfolio_returns.loc[regime_dates]

        summary = summarize_risk_metrics(
            regime_returns,
            risk_free_rate=risk_free_rate,
            confidence_level=confidence_level,
            annualization_factor=annualization_factor,
        )

        rows.append(
            {
                "regime": int(regime),
                "observation_count": int(len(regime_returns)),
                **summary,
            }
        )

    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def calculate_regime_asset_risk_summary(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    risk_free_rate: float = 0.0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate asset-level risk metrics grouped by regime."""
    _validate_returns(returns, return_col=return_col)

    wide_returns = _pivot_returns(returns, return_col=return_col)
    labels = _prepare_regime_labels(regime_labels)

    _, aligned_labels = _align_returns_and_labels(
        wide_returns.mean(axis=1),
        labels,
    )
    aligned_returns = wide_returns.loc[aligned_labels.index]

    rows: list[dict[str, object]] = []

    for regime in sorted(aligned_labels.unique()):
        regime_dates = aligned_labels[aligned_labels == regime].index
        regime_returns = aligned_returns.loc[regime_dates]

        for ticker in regime_returns.columns:
            ticker_returns = regime_returns[ticker]

            summary = summarize_risk_metrics(
                ticker_returns,
                risk_free_rate=risk_free_rate,
                confidence_level=confidence_level,
                annualization_factor=annualization_factor,
            )

            rows.append(
                {
                    "regime": int(regime),
                    "ticker": str(ticker),
                    "observation_count": int(len(ticker_returns)),
                    **summary,
                }
            )

    return pd.DataFrame(rows).sort_values(["regime", "ticker"]).reset_index(drop=True)


def build_regime_risk_summary(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    weights: Mapping[str, float] | None = None,
    risk_free_rate: float = 0.0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    annualization_factor: int = DEFAULT_ANNUALIZATION_FACTOR,
    return_col: str = "return",
) -> RegimeRiskSummary:
    """Build portfolio-level and asset-level regime risk summaries."""
    portfolio_summary = calculate_regime_portfolio_risk_summary(
        returns=returns,
        regime_labels=regime_labels,
        weights=weights,
        risk_free_rate=risk_free_rate,
        confidence_level=confidence_level,
        annualization_factor=annualization_factor,
        return_col=return_col,
    )
    asset_summary = calculate_regime_asset_risk_summary(
        returns=returns,
        regime_labels=regime_labels,
        risk_free_rate=risk_free_rate,
        confidence_level=confidence_level,
        annualization_factor=annualization_factor,
        return_col=return_col,
    )

    return RegimeRiskSummary(
        portfolio_summary=portfolio_summary,
        asset_summary=asset_summary,
    )


def _validate_returns(returns: pd.DataFrame, return_col: str) -> None:
    required_columns = {"date", "ticker", return_col}
    missing_columns = required_columns.difference(returns.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RegimeRiskSummaryError(f"Missing required return column(s): {missing}")

    if returns.empty:
        raise RegimeRiskSummaryError("Return data is empty")

    if returns[return_col].isna().any():
        raise RegimeRiskSummaryError("Return data contains missing values")

    duplicate_rows = returns.duplicated(subset=["date", "ticker"])

    if duplicate_rows.any():
        raise RegimeRiskSummaryError("Return data contains duplicate date/ticker rows")


def _prepare_regime_labels(regime_labels: pd.Series) -> pd.Series:
    if not isinstance(regime_labels.index, pd.DatetimeIndex):
        raise RegimeRiskSummaryError("Regime labels index must be a DatetimeIndex")

    if regime_labels.empty:
        raise RegimeRiskSummaryError("Regime labels are empty")

    if regime_labels.index.has_duplicates:
        raise RegimeRiskSummaryError("Regime labels contain duplicate dates")

    if regime_labels.isna().any():
        raise RegimeRiskSummaryError("Regime labels contain missing values")

    labels = regime_labels.copy()
    labels.index = pd.to_datetime(labels.index)

    return labels.sort_index()


def _pivot_returns(returns: pd.DataFrame, return_col: str) -> pd.DataFrame:
    normalized_returns = returns[["date", "ticker", return_col]].copy()
    normalized_returns["date"] = pd.to_datetime(normalized_returns["date"])
    normalized_returns["ticker"] = (
        normalized_returns["ticker"].astype(str).str.upper().str.strip()
    )
    normalized_returns[return_col] = pd.to_numeric(
        normalized_returns[return_col],
        errors="coerce",
    )

    wide_returns = normalized_returns.pivot(
        index="date",
        columns="ticker",
        values=return_col,
    )

    if wide_returns.isna().any().any():
        raise RegimeRiskSummaryError("Return data contains incomplete ticker coverage")

    return wide_returns.sort_index()


def _validate_weights(
    weights: Mapping[str, float],
    tickers: list[str],
) -> dict[str, float]:
    if not weights:
        raise RegimeRiskSummaryError("Portfolio weights cannot be empty")

    clean_weights = {
        ticker.upper().strip(): float(weight) for ticker, weight in weights.items()
    }

    expected_tickers = set(tickers)
    weight_tickers = set(clean_weights)

    missing_tickers = sorted(expected_tickers.difference(weight_tickers))
    extra_tickers = sorted(weight_tickers.difference(expected_tickers))

    if missing_tickers:
        missing = ", ".join(missing_tickers)
        raise RegimeRiskSummaryError(f"Missing weight(s) for ticker(s): {missing}")

    if extra_tickers:
        extra = ", ".join(extra_tickers)
        raise RegimeRiskSummaryError(f"Weight provided for unknown ticker(s): {extra}")

    if any(weight < 0 for weight in clean_weights.values()):
        raise RegimeRiskSummaryError("Portfolio weights must be non-negative")

    weight_sum = sum(clean_weights.values())

    if abs(weight_sum - 1.0) > 1e-8:
        raise RegimeRiskSummaryError("Portfolio weights must sum to 1.0")

    return clean_weights


def _align_returns_and_labels(
    returns: pd.Series,
    regime_labels: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    overlapping_dates = returns.index.intersection(regime_labels.index)

    if overlapping_dates.empty:
        raise RegimeRiskSummaryError(
            "Returns and regime labels have no overlapping dates"
        )

    aligned_returns = returns.loc[overlapping_dates].sort_index()
    aligned_labels = regime_labels.loc[overlapping_dates].sort_index()

    return aligned_returns, aligned_labels
