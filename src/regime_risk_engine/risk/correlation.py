from dataclasses import dataclass

import numpy as np
import pandas as pd


class RegimeCorrelationError(ValueError):
    """Raised when regime correlation analytics cannot be calculated."""


@dataclass(frozen=True, slots=True)
class RegimeCorrelationAnalytics:
    """Container for regime correlation and covariance analytics."""

    correlation_matrices: dict[int, pd.DataFrame]
    covariance_matrices: dict[int, pd.DataFrame]
    summary: pd.DataFrame


def calculate_regime_correlation_matrices(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    return_col: str = "return",
) -> dict[int, pd.DataFrame]:
    """Calculate asset correlation matrix for each regime."""
    wide_returns, labels = _prepare_aligned_returns_and_labels(
        returns=returns,
        regime_labels=regime_labels,
        return_col=return_col,
    )

    matrices: dict[int, pd.DataFrame] = {}

    for regime in sorted(labels.unique()):
        regime_dates = labels[labels == regime].index
        regime_returns = wide_returns.loc[regime_dates]
        matrices[int(regime)] = regime_returns.corr()

    return matrices


def calculate_regime_covariance_matrices(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    annualization_factor: int = 252,
    return_col: str = "return",
) -> dict[int, pd.DataFrame]:
    """Calculate annualized asset covariance matrix for each regime."""
    if annualization_factor <= 0:
        raise RegimeCorrelationError("Annualization factor must be positive")

    wide_returns, labels = _prepare_aligned_returns_and_labels(
        returns=returns,
        regime_labels=regime_labels,
        return_col=return_col,
    )

    matrices: dict[int, pd.DataFrame] = {}

    for regime in sorted(labels.unique()):
        regime_dates = labels[labels == regime].index
        regime_returns = wide_returns.loc[regime_dates]
        matrices[int(regime)] = regime_returns.cov() * annualization_factor

    return matrices


def calculate_regime_average_correlation_summary(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate average pairwise correlation by regime."""
    correlation_matrices = calculate_regime_correlation_matrices(
        returns=returns,
        regime_labels=regime_labels,
        return_col=return_col,
    )

    rows: list[dict[str, float | int]] = []

    for regime, matrix in correlation_matrices.items():
        rows.append(
            {
                "regime": regime,
                "average_pairwise_correlation": _average_off_diagonal_correlation(
                    matrix
                ),
                "asset_count": int(matrix.shape[0]),
            }
        )

    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def build_regime_correlation_analytics(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    annualization_factor: int = 252,
    return_col: str = "return",
) -> RegimeCorrelationAnalytics:
    """Build regime correlation matrices, covariance matrices, and summary."""
    correlation_matrices = calculate_regime_correlation_matrices(
        returns=returns,
        regime_labels=regime_labels,
        return_col=return_col,
    )
    covariance_matrices = calculate_regime_covariance_matrices(
        returns=returns,
        regime_labels=regime_labels,
        annualization_factor=annualization_factor,
        return_col=return_col,
    )
    summary = calculate_regime_average_correlation_summary(
        returns=returns,
        regime_labels=regime_labels,
        return_col=return_col,
    )

    return RegimeCorrelationAnalytics(
        correlation_matrices=correlation_matrices,
        covariance_matrices=covariance_matrices,
        summary=summary,
    )


def _prepare_aligned_returns_and_labels(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    return_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    _validate_returns(returns, return_col=return_col)
    labels = _validate_and_prepare_regime_labels(regime_labels)
    wide_returns = _pivot_returns(returns, return_col=return_col)

    overlapping_dates = wide_returns.index.intersection(labels.index)

    if overlapping_dates.empty:
        raise RegimeCorrelationError(
            "Returns and regime labels have no overlapping dates"
        )

    aligned_returns = wide_returns.loc[overlapping_dates].sort_index()
    aligned_labels = labels.loc[overlapping_dates].sort_index()

    return aligned_returns, aligned_labels


def _validate_returns(returns: pd.DataFrame, return_col: str) -> None:
    required_columns = {"date", "ticker", return_col}
    missing_columns = required_columns.difference(returns.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RegimeCorrelationError(f"Missing required return column(s): {missing}")

    if returns.empty:
        raise RegimeCorrelationError("Return data is empty")

    if returns[return_col].isna().any():
        raise RegimeCorrelationError("Return data contains missing values")

    duplicate_rows = returns.duplicated(subset=["date", "ticker"])

    if duplicate_rows.any():
        raise RegimeCorrelationError("Return data contains duplicate date/ticker rows")


def _validate_and_prepare_regime_labels(regime_labels: pd.Series) -> pd.Series:
    if not isinstance(regime_labels.index, pd.DatetimeIndex):
        raise RegimeCorrelationError("Regime labels index must be a DatetimeIndex")

    if regime_labels.empty:
        raise RegimeCorrelationError("Regime labels are empty")

    if regime_labels.index.has_duplicates:
        raise RegimeCorrelationError("Regime labels contain duplicate dates")

    if regime_labels.isna().any():
        raise RegimeCorrelationError("Regime labels contain missing values")

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
    ).sort_index()

    if wide_returns.isna().any().any():
        raise RegimeCorrelationError("Return data contains incomplete ticker coverage")

    if wide_returns.shape[1] < 2:
        raise RegimeCorrelationError("At least two assets are required")

    return wide_returns


def _average_off_diagonal_correlation(correlation_matrix: pd.DataFrame) -> float:
    values = correlation_matrix.to_numpy(dtype=float)
    mask = ~np.eye(values.shape[0], dtype=bool)

    off_diagonal_values = [
        float(value) for value in values[mask] if not np.isnan(float(value))
    ]

    if not off_diagonal_values:
        return float("nan")

    return sum(off_diagonal_values) / len(off_diagonal_values)
