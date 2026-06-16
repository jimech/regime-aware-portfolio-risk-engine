from collections.abc import Mapping

import numpy as np
import pandas as pd


class RegimeLabelingError(ValueError):
    """Raised when regime labeling cannot be completed."""


REQUIRED_RETURN_COLUMNS = {"date", "ticker", "return"}

SUMMARY_COLUMNS = [
    "regime",
    "observation_count",
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "average_pairwise_correlation",
]


def calculate_regime_summary(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    annualization_factor: int = 252,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate diagnostic statistics for each numeric regime.

    Args:
        returns: Long-format return data with date, ticker, and return columns.
        regime_labels: Date-indexed numeric regime labels.
        annualization_factor: Trading periods per year.
        return_col: Name of the return column.

    Returns:
        Regime summary DataFrame.
    """
    _validate_returns(returns, return_col=return_col)
    _validate_regime_labels(regime_labels)

    if annualization_factor <= 0:
        raise RegimeLabelingError("Annualization factor must be positive")

    normalized_returns = _prepare_returns(returns, return_col=return_col)
    normalized_labels = _prepare_regime_labels(regime_labels)

    overlapping_dates = normalized_returns.index.intersection(normalized_labels.index)

    if overlapping_dates.empty:
        raise RegimeLabelingError("Returns and regime labels have no overlapping dates")

    normalized_returns = normalized_returns.loc[overlapping_dates]
    normalized_labels = normalized_labels.loc[overlapping_dates]

    portfolio_returns = normalized_returns.mean(axis=1)
    summary_rows: list[dict[str, float | int]] = []

    for regime in sorted(normalized_labels.unique()):
        regime_dates = normalized_labels[normalized_labels == regime].index
        regime_portfolio_returns = portfolio_returns.loc[regime_dates]
        regime_asset_returns = normalized_returns.loc[regime_dates]

        summary_rows.append(
            {
                "regime": int(regime),
                "observation_count": int(len(regime_dates)),
                "annualized_return": _annualized_return(
                    regime_portfolio_returns,
                    annualization_factor=annualization_factor,
                ),
                "annualized_volatility": _annualized_volatility(
                    regime_portfolio_returns,
                    annualization_factor=annualization_factor,
                ),
                "max_drawdown": _max_drawdown(regime_portfolio_returns),
                "average_pairwise_correlation": _average_pairwise_correlation(
                    regime_asset_returns
                ),
            }
        )

    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def assign_regime_labels(
    regime_summary: pd.DataFrame,
) -> dict[int, str]:
    """Assign human-readable labels to numeric regimes using transparent rules."""
    _validate_regime_summary(regime_summary)

    summary = regime_summary.copy()

    median_return = float(summary["annualized_return"].median())
    median_volatility = float(summary["annualized_volatility"].median())

    highest_vol_regime = int(
        summary.loc[summary["annualized_volatility"].idxmax(), "regime"]
    )
    best_return_regime = int(
        summary.loc[summary["annualized_return"].idxmax(), "regime"]
    )

    label_map: dict[int, str] = {}

    for row in summary.to_dict(orient="records"):
        regime = int(row["regime"])
        annualized_return = float(row["annualized_return"])
        annualized_volatility = float(row["annualized_volatility"])
        max_drawdown = float(row["max_drawdown"])

        if regime == highest_vol_regime and max_drawdown < 0:
            label = "high_volatility_stress"
        elif (
            regime == best_return_regime and annualized_volatility <= median_volatility
        ):
            label = "bull_low_volatility"
        elif regime == best_return_regime:
            label = "growth_recovery"
        elif (
            annualized_return < median_return
            and annualized_volatility <= median_volatility
        ):
            label = "defensive_low_return"
        elif annualized_return < median_return:
            label = "bear_or_drawdown"
        else:
            label = "neutral_mixed"

        label_map[regime] = label

    return label_map


def apply_regime_label_map(
    regime_labels: pd.Series,
    label_map: Mapping[int, str],
) -> pd.Series:
    """Convert numeric regime labels into human-readable labels."""
    _validate_regime_labels(regime_labels)

    missing_labels = sorted(
        set(int(regime) for regime in regime_labels.unique()).difference(label_map)
    )

    if missing_labels:
        missing = ", ".join(str(regime) for regime in missing_labels)
        raise RegimeLabelingError(
            f"Missing human-readable label(s) for regime: {missing}"
        )

    readable_labels = regime_labels.map(lambda regime: label_map[int(regime)])

    return pd.Series(
        readable_labels,
        index=regime_labels.index,
        name="regime_label",
    )


def build_labeled_regime_summary(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    annualization_factor: int = 252,
    return_col: str = "return",
) -> pd.DataFrame:
    """Build a regime summary with human-readable labels."""
    summary = calculate_regime_summary(
        returns=returns,
        regime_labels=regime_labels,
        annualization_factor=annualization_factor,
        return_col=return_col,
    )
    label_map = assign_regime_labels(summary)

    labeled_summary = summary.copy()
    labeled_summary["regime_label"] = labeled_summary["regime"].map(label_map)

    return labeled_summary


def _validate_returns(returns: pd.DataFrame, return_col: str) -> None:
    required_columns = {"date", "ticker", return_col}
    missing_columns = required_columns.difference(returns.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RegimeLabelingError(f"Missing required return column(s): {missing}")

    if returns.empty:
        raise RegimeLabelingError("Return data is empty")

    if returns[return_col].isna().any():
        raise RegimeLabelingError("Return data contains missing values")

    duplicate_rows = returns.duplicated(subset=["date", "ticker"])

    if duplicate_rows.any():
        raise RegimeLabelingError("Return data contains duplicate date/ticker rows")


def _validate_regime_labels(regime_labels: pd.Series) -> None:
    if not isinstance(regime_labels.index, pd.DatetimeIndex):
        raise RegimeLabelingError("Regime labels index must be a DatetimeIndex")

    if regime_labels.empty:
        raise RegimeLabelingError("Regime labels are empty")

    if regime_labels.index.has_duplicates:
        raise RegimeLabelingError("Regime labels contain duplicate dates")

    if regime_labels.isna().any():
        raise RegimeLabelingError("Regime labels contain missing values")


def _validate_regime_summary(regime_summary: pd.DataFrame) -> None:
    missing_columns = set(SUMMARY_COLUMNS).difference(regime_summary.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RegimeLabelingError(f"Missing regime summary column(s): {missing}")

    if regime_summary.empty:
        raise RegimeLabelingError("Regime summary is empty")

    if regime_summary["regime"].duplicated().any():
        raise RegimeLabelingError("Regime summary contains duplicate regimes")


def _prepare_returns(returns: pd.DataFrame, return_col: str) -> pd.DataFrame:
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

    return wide_returns.sort_index()


def _prepare_regime_labels(regime_labels: pd.Series) -> pd.Series:
    labels = regime_labels.copy()
    labels.index = pd.to_datetime(labels.index)

    return labels.sort_index()


def _annualized_return(
    returns: pd.Series,
    annualization_factor: int,
) -> float:
    if returns.empty:
        return float("nan")

    compounded_return = float((1.0 + returns).prod())
    periods = len(returns)

    return float(compounded_return ** (annualization_factor / periods) - 1.0)


def _annualized_volatility(
    returns: pd.Series,
    annualization_factor: int,
) -> float:
    if len(returns) < 2:
        return float("nan")

    return float(returns.std(ddof=1) * annualization_factor**0.5)


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")

    return_index = (1.0 + returns).cumprod()
    running_peak = return_index.cummax()
    drawdown = return_index / running_peak - 1.0

    return float(drawdown.min())


def _average_pairwise_correlation(returns: pd.DataFrame) -> float:
    if returns.shape[1] < 2:
        return float("nan")

    correlation_matrix = returns.corr()
    values = correlation_matrix.to_numpy(dtype=float)
    mask = ~np.eye(values.shape[0], dtype=bool)

    off_diagonal_values = [
        float(value) for value in values[mask] if not np.isnan(float(value))
    ]

    if not off_diagonal_values:
        return float("nan")

    return sum(off_diagonal_values) / len(off_diagonal_values)
