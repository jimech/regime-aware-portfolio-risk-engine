from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_TREND_WINDOWS = {
    "short": 21,
    "medium": 63,
    "long": 126,
    "annual": 252,
}


class TrendFeatureError(ValueError):
    """Raised when trend features cannot be calculated."""


def calculate_momentum_features(
    returns: pd.DataFrame,
    windows: Mapping[str, int] | None = None,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate rolling momentum features by ticker."""
    _validate_returns_data(returns, return_col=return_col)
    selected_windows = _validate_windows(windows or DEFAULT_TREND_WINDOWS)

    sorted_returns = _prepare_returns(returns, return_col=return_col)
    result = sorted_returns[["date", "ticker"]].copy()

    grouped_returns = sorted_returns.groupby("ticker", group_keys=False)[return_col]

    for window_name, window_size in selected_windows.items():
        feature_name = f"momentum_{window_name}_{window_size}d"
        result[feature_name] = grouped_returns.transform(
            lambda series, size=window_size: (
                series.rolling(size).apply(
                    _compound_simple_returns,
                    raw=True,
                )
                - 1.0
            )
        )

    return result


def calculate_moving_average_distance_features(
    returns: pd.DataFrame,
    windows: Mapping[str, int] | None = None,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate distance from moving average by ticker."""
    _validate_returns_data(returns, return_col=return_col)
    selected_windows = _validate_windows(windows or DEFAULT_TREND_WINDOWS)

    sorted_returns = _prepare_returns(returns, return_col=return_col)
    sorted_returns["return_index"] = _calculate_return_index(sorted_returns, return_col)

    result = sorted_returns[["date", "ticker"]].copy()
    grouped_index = sorted_returns.groupby("ticker", group_keys=False)["return_index"]

    for window_name, window_size in selected_windows.items():
        feature_name = f"ma_distance_{window_name}_{window_size}d"
        result[feature_name] = grouped_index.transform(
            lambda series, size=window_size: series / series.rolling(size).mean() - 1.0
        )

    return result


def calculate_drawdown_features(
    returns: pd.DataFrame,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate drawdown from cumulative return index by ticker."""
    _validate_returns_data(returns, return_col=return_col)

    sorted_returns = _prepare_returns(returns, return_col=return_col)
    sorted_returns["return_index"] = _calculate_return_index(sorted_returns, return_col)

    grouped_index = sorted_returns.groupby("ticker", group_keys=False)["return_index"]
    running_peak = grouped_index.transform("cummax")

    result = sorted_returns[["date", "ticker"]].copy()
    result["drawdown"] = sorted_returns["return_index"] / running_peak - 1.0

    return result


def build_trend_features(
    returns: pd.DataFrame,
    windows: Mapping[str, int] | None = None,
    return_col: str = "return",
) -> pd.DataFrame:
    """Build momentum, moving-average distance, and drawdown features."""
    momentum = calculate_momentum_features(
        returns=returns,
        windows=windows,
        return_col=return_col,
    )
    moving_average_distance = calculate_moving_average_distance_features(
        returns=returns,
        windows=windows,
        return_col=return_col,
    )
    drawdown = calculate_drawdown_features(
        returns=returns,
        return_col=return_col,
    )

    return momentum.merge(
        moving_average_distance,
        on=["date", "ticker"],
        how="inner",
    ).merge(
        drawdown,
        on=["date", "ticker"],
        how="inner",
    )


def _validate_returns_data(returns: pd.DataFrame, return_col: str) -> None:
    required_columns = {"date", "ticker", return_col}
    missing_columns = required_columns.difference(returns.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise TrendFeatureError(f"Missing required return column(s): {missing}")

    if returns.empty:
        raise TrendFeatureError("Return data is empty")

    if returns[return_col].isna().any():
        raise TrendFeatureError("Return data contains missing values")


def _validate_windows(windows: Mapping[str, int]) -> dict[str, int]:
    if not windows:
        raise TrendFeatureError("At least one trend window is required")

    validated_windows: dict[str, int] = {}

    for window_name, window_size in windows.items():
        if not isinstance(window_name, str) or not window_name.strip():
            raise TrendFeatureError("Trend window names must be non-empty")

        if window_size <= 1:
            raise TrendFeatureError("Trend window sizes must be greater than 1")

        validated_windows[window_name] = window_size

    return validated_windows


def _prepare_returns(returns: pd.DataFrame, return_col: str) -> pd.DataFrame:
    sorted_returns = returns[["date", "ticker", return_col]].copy()
    sorted_returns["date"] = pd.to_datetime(sorted_returns["date"])
    sorted_returns["ticker"] = (
        sorted_returns["ticker"].astype(str).str.upper().str.strip()
    )

    return sorted_returns.sort_values(["ticker", "date"]).reset_index(drop=True)


def _calculate_return_index(returns: pd.DataFrame, return_col: str) -> pd.Series:
    gross_returns = 1.0 + returns[return_col]

    return gross_returns.groupby(returns["ticker"]).cumprod()


def _compound_simple_returns(values: Any) -> float:
    return float(np.prod(1.0 + values))
