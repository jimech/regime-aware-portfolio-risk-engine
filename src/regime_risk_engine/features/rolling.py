from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ROLLING_WINDOWS = {
    "short": 21,
    "medium": 63,
    "long": 126,
    "annual": 252,
}

REQUIRED_RETURN_COLUMNS = {"date", "ticker", "return"}


class FeatureEngineeringError(ValueError):
    """Raised when features cannot be calculated."""


def calculate_rolling_returns(
    returns: pd.DataFrame,
    windows: Mapping[str, int] | None = None,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate rolling cumulative simple returns by ticker."""
    _validate_returns_data(returns, return_col=return_col)
    selected_windows = _validate_windows(windows or DEFAULT_ROLLING_WINDOWS)

    sorted_returns = _prepare_returns(returns, return_col=return_col)
    result = sorted_returns[["date", "ticker"]].copy()

    grouped_returns = sorted_returns.groupby("ticker", group_keys=False)[return_col]

    for window_name, window_size in selected_windows.items():
        feature_name = f"rolling_return_{window_name}_{window_size}d"
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


def calculate_rolling_volatility(
    returns: pd.DataFrame,
    windows: Mapping[str, int] | None = None,
    annualization_factor: int = 252,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate annualized rolling volatility by ticker."""
    _validate_returns_data(returns, return_col=return_col)
    selected_windows = _validate_windows(windows or DEFAULT_ROLLING_WINDOWS)

    if annualization_factor <= 0:
        raise FeatureEngineeringError("Annualization factor must be positive")

    sorted_returns = _prepare_returns(returns, return_col=return_col)
    result = sorted_returns[["date", "ticker"]].copy()

    grouped_returns = sorted_returns.groupby("ticker", group_keys=False)[return_col]

    for window_name, window_size in selected_windows.items():
        feature_name = f"rolling_volatility_{window_name}_{window_size}d"
        result[feature_name] = grouped_returns.transform(
            lambda series, size=window_size: (
                series.rolling(size).std() * annualization_factor**0.5
            )
        )

    return result


def build_rolling_features(
    returns: pd.DataFrame,
    windows: Mapping[str, int] | None = None,
    annualization_factor: int = 252,
    return_col: str = "return",
) -> pd.DataFrame:
    """Build rolling return and volatility features."""
    rolling_returns = calculate_rolling_returns(
        returns=returns,
        windows=windows,
        return_col=return_col,
    )
    rolling_volatility = calculate_rolling_volatility(
        returns=returns,
        windows=windows,
        annualization_factor=annualization_factor,
        return_col=return_col,
    )

    return rolling_returns.merge(
        rolling_volatility,
        on=["date", "ticker"],
        how="inner",
    )


def _validate_returns_data(returns: pd.DataFrame, return_col: str) -> None:
    required_columns = {"date", "ticker", return_col}
    missing_columns = required_columns.difference(returns.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise FeatureEngineeringError(f"Missing required return column(s): {missing}")

    if returns.empty:
        raise FeatureEngineeringError("Return data is empty")

    if returns[return_col].isna().any():
        raise FeatureEngineeringError("Return data contains missing values")


def _validate_windows(windows: Mapping[str, int]) -> dict[str, int]:
    if not windows:
        raise FeatureEngineeringError("At least one rolling window is required")

    validated_windows: dict[str, int] = {}

    for window_name, window_size in windows.items():
        if not isinstance(window_name, str) or not window_name.strip():
            raise FeatureEngineeringError("Rolling window names must be non-empty")

        if window_size <= 1:
            raise FeatureEngineeringError("Rolling window sizes must be greater than 1")

        validated_windows[window_name] = window_size

    return validated_windows


def _prepare_returns(returns: pd.DataFrame, return_col: str) -> pd.DataFrame:
    sorted_returns = returns[["date", "ticker", return_col]].copy()
    sorted_returns["date"] = pd.to_datetime(sorted_returns["date"])
    sorted_returns["ticker"] = (
        sorted_returns["ticker"].astype(str).str.upper().str.strip()
    )

    return sorted_returns.sort_values(["ticker", "date"]).reset_index(drop=True)


def _compound_simple_returns(values: Any) -> float:
    return float(np.prod(1.0 + values))
