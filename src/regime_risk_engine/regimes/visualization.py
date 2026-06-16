from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

REQUIRED_RETURN_COLUMNS = {"date", "ticker", "return"}


class RegimeVisualizationError(ValueError):
    """Raised when regime visualizations cannot be created."""


def plot_regime_timeline(
    regime_labels: pd.Series,
    label_map: Mapping[int, str] | None = None,
    title: str = "Detected Market Regimes",
) -> Figure:
    """Plot regime labels over time.

    Args:
        regime_labels: Date-indexed numeric regime labels.
        label_map: Optional mapping from numeric labels to readable labels.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    _validate_regime_labels(regime_labels)

    labels = _prepare_regime_labels(regime_labels)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(labels.index, labels.values, where="post")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Regime")
    ax.grid(True, alpha=0.3)

    if label_map:
        regimes = sorted(labels.unique())
        ax.set_yticks(regimes)
        ax.set_yticklabels(
            [label_map.get(int(regime), str(regime)) for regime in regimes]
        )

    fig.tight_layout()

    return fig


def plot_regime_probabilities(
    probabilities: pd.DataFrame,
    title: str = "Regime Probabilities",
) -> Figure:
    """Plot regime probabilities over time.

    Args:
        probabilities: Date-indexed DataFrame of regime probabilities.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    _validate_probability_frame(probabilities)

    fig, ax = plt.subplots(figsize=(12, 4))

    for column in probabilities.columns:
        ax.plot(probabilities.index, probabilities[column], label=column)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_cumulative_returns_with_regimes(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    title: str = "Cumulative Returns by Regime",
    return_col: str = "return",
) -> Figure:
    """Plot equal-weight cumulative portfolio returns with regime shading.

    Args:
        returns: Long-format returns with date, ticker, and return columns.
        regime_labels: Date-indexed numeric regime labels.
        title: Plot title.
        return_col: Name of the return column.

    Returns:
        Matplotlib Figure.
    """
    _validate_returns(returns, return_col=return_col)
    _validate_regime_labels(regime_labels)

    portfolio_returns = _calculate_equal_weight_portfolio_returns(
        returns,
        return_col=return_col,
    )
    labels = _prepare_regime_labels(regime_labels)

    overlapping_dates = portfolio_returns.index.intersection(labels.index)

    if overlapping_dates.empty:
        raise RegimeVisualizationError(
            "Returns and regime labels have no overlapping dates"
        )

    portfolio_returns = portfolio_returns.loc[overlapping_dates]
    labels = labels.loc[overlapping_dates]
    cumulative_returns = (1.0 + portfolio_returns).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cumulative_returns.index, cumulative_returns.values)
    _shade_regime_background(ax, labels)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def _shade_regime_background(ax: Axes, regime_labels: pd.Series) -> None:
    labels = _prepare_regime_labels(regime_labels)

    if labels.empty:
        return

    start_date = labels.index[0]
    previous_date = labels.index[0]
    current_regime = labels.iloc[0]

    for date, regime in labels.iloc[1:].items():
        if regime != current_regime:
            ax.axvspan(start_date, previous_date, alpha=0.12)
            start_date = date
            current_regime = regime

        previous_date = date

    ax.axvspan(start_date, previous_date, alpha=0.12)


def _calculate_equal_weight_portfolio_returns(
    returns: pd.DataFrame,
    return_col: str,
) -> pd.Series:
    normalized_returns = returns[["date", "ticker", return_col]].copy()
    normalized_returns["date"] = pd.to_datetime(normalized_returns["date"])
    normalized_returns["ticker"] = (
        normalized_returns["ticker"].astype(str).str.upper().str.strip()
    )
    normalized_returns[return_col] = pd.to_numeric(
        normalized_returns[return_col],
        errors="coerce",
    )

    portfolio_returns = normalized_returns.pivot(
        index="date",
        columns="ticker",
        values=return_col,
    ).mean(axis=1)

    portfolio_returns.name = "portfolio_return"

    return portfolio_returns.sort_index()


def _validate_returns(returns: pd.DataFrame, return_col: str) -> None:
    required_columns = {"date", "ticker", return_col}
    missing_columns = required_columns.difference(returns.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RegimeVisualizationError(f"Missing required return column(s): {missing}")

    if returns.empty:
        raise RegimeVisualizationError("Return data is empty")

    if returns[return_col].isna().any():
        raise RegimeVisualizationError("Return data contains missing values")

    duplicate_rows = returns.duplicated(subset=["date", "ticker"])

    if duplicate_rows.any():
        raise RegimeVisualizationError(
            "Return data contains duplicate date/ticker rows"
        )


def _validate_regime_labels(regime_labels: pd.Series) -> None:
    if not isinstance(regime_labels.index, pd.DatetimeIndex):
        raise RegimeVisualizationError("Regime labels index must be a DatetimeIndex")

    if regime_labels.empty:
        raise RegimeVisualizationError("Regime labels are empty")

    if regime_labels.index.has_duplicates:
        raise RegimeVisualizationError("Regime labels contain duplicate dates")

    if regime_labels.isna().any():
        raise RegimeVisualizationError("Regime labels contain missing values")


def _validate_probability_frame(probabilities: pd.DataFrame) -> None:
    if not isinstance(probabilities.index, pd.DatetimeIndex):
        raise RegimeVisualizationError("Probability index must be a DatetimeIndex")

    if probabilities.empty:
        raise RegimeVisualizationError("Probability data is empty")

    if probabilities.index.has_duplicates:
        raise RegimeVisualizationError("Probability data contains duplicate dates")

    if probabilities.isna().any().any():
        raise RegimeVisualizationError("Probability data contains missing values")

    row_sums = probabilities.sum(axis=1)

    if not row_sums.between(0.999, 1.001).all():
        raise RegimeVisualizationError("Probability rows must sum to 1.0")


def _prepare_regime_labels(regime_labels: pd.Series) -> pd.Series:
    labels = regime_labels.copy()
    labels.index = pd.to_datetime(labels.index)

    return labels.sort_index()
