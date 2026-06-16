from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_CORRELATION_WINDOWS = {
    "short": 21,
    "medium": 63,
    "long": 126,
    "annual": 252,
}


class CorrelationFeatureError(ValueError):
    """Raised when correlation features cannot be calculated."""


def calculate_rolling_correlation_matrices(
    returns: pd.DataFrame,
    window: int,
    return_col: str = "return",
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Calculate rolling correlation matrices by date."""
    _validate_returns_data(returns, return_col=return_col)

    if window <= 1:
        raise CorrelationFeatureError("Correlation window must be greater than 1")

    wide_returns = _pivot_returns(returns, return_col=return_col)
    matrices: dict[pd.Timestamp, pd.DataFrame] = {}

    for end_position, date in enumerate(wide_returns.index):
        if end_position + 1 < window:
            continue

        start_position = end_position + 1 - window
        window_returns = wide_returns.iloc[start_position : end_position + 1]
        matrices[pd.Timestamp(date)] = window_returns.corr()

    return matrices


def calculate_rolling_average_correlation(
    returns: pd.DataFrame,
    windows: Mapping[str, int] | None = None,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate average pairwise rolling correlation across all tickers."""
    _validate_returns_data(returns, return_col=return_col)
    selected_windows = _validate_windows(windows or DEFAULT_CORRELATION_WINDOWS)

    wide_returns = _pivot_returns(returns, return_col=return_col)
    result = pd.DataFrame({"date": wide_returns.index})

    for window_name, window_size in selected_windows.items():
        feature_name = f"avg_pairwise_corr_{window_name}_{window_size}d"
        values: list[float] = []

        for end_position in range(len(wide_returns)):
            if end_position + 1 < window_size:
                values.append(np.nan)
                continue

            start_position = end_position + 1 - window_size
            window_returns = wide_returns.iloc[start_position : end_position + 1]
            correlation_matrix = window_returns.corr()
            values.append(_average_off_diagonal_correlation(correlation_matrix))

        result[feature_name] = values

    return result


def calculate_rolling_pairwise_correlations(
    returns: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    windows: Mapping[str, int] | None = None,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate rolling correlations for selected ticker pairs."""
    _validate_returns_data(returns, return_col=return_col)
    selected_windows = _validate_windows(windows or DEFAULT_CORRELATION_WINDOWS)
    clean_pairs = _validate_pairs(pairs)

    wide_returns = _pivot_returns(returns, return_col=return_col)
    _validate_pair_tickers_exist(wide_returns, clean_pairs)

    result = pd.DataFrame({"date": wide_returns.index})

    for ticker_a, ticker_b in clean_pairs:
        for window_name, window_size in selected_windows.items():
            feature_name = (
                f"rolling_corr_{ticker_a}_{ticker_b}_{window_name}_{window_size}d"
            )
            result[feature_name] = _calculate_pairwise_rolling_correlation(
                wide_returns=wide_returns,
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                window_size=window_size,
            )

    return result


def calculate_equity_bond_correlation(
    returns: pd.DataFrame,
    equity_ticker: str = "SPY",
    bond_ticker: str = "TLT",
    windows: Mapping[str, int] | None = None,
    return_col: str = "return",
) -> pd.DataFrame:
    """Calculate rolling equity-bond correlation features."""
    return calculate_rolling_pairwise_correlations(
        returns=returns,
        pairs=[(equity_ticker, bond_ticker)],
        windows=windows,
        return_col=return_col,
    )


def build_correlation_features(
    returns: pd.DataFrame,
    pairs: Sequence[tuple[str, str]] | None = None,
    equity_ticker: str = "SPY",
    bond_ticker: str = "TLT",
    windows: Mapping[str, int] | None = None,
    return_col: str = "return",
) -> pd.DataFrame:
    """Build average pairwise and selected pairwise correlation features."""
    average_correlation = calculate_rolling_average_correlation(
        returns=returns,
        windows=windows,
        return_col=return_col,
    )

    selected_pairs = pairs or [(equity_ticker, bond_ticker)]
    pairwise_correlation = calculate_rolling_pairwise_correlations(
        returns=returns,
        pairs=selected_pairs,
        windows=windows,
        return_col=return_col,
    )

    return average_correlation.merge(
        pairwise_correlation,
        on="date",
        how="inner",
    )


def _calculate_pairwise_rolling_correlation(
    wide_returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window_size: int,
) -> list[float]:
    values: list[float] = []

    for end_position in range(len(wide_returns)):
        if end_position + 1 < window_size:
            values.append(np.nan)
            continue

        start_position = end_position + 1 - window_size
        window_returns = wide_returns.iloc[start_position : end_position + 1]
        pair_returns = window_returns[[ticker_a, ticker_b]].dropna()

        if len(pair_returns) < window_size:
            values.append(np.nan)
            continue

        correlation = pair_returns[ticker_a].corr(pair_returns[ticker_b])
        values.append(float(correlation))

    return values


def _validate_returns_data(returns: pd.DataFrame, return_col: str) -> None:
    required_columns = {"date", "ticker", return_col}
    missing_columns = required_columns.difference(returns.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise CorrelationFeatureError(f"Missing required return column(s): {missing}")

    if returns.empty:
        raise CorrelationFeatureError("Return data is empty")

    if returns[return_col].isna().any():
        raise CorrelationFeatureError("Return data contains missing values")

    duplicate_rows = returns.duplicated(subset=["date", "ticker"])

    if duplicate_rows.any():
        raise CorrelationFeatureError("Return data contains duplicate date/ticker rows")


def _validate_windows(windows: Mapping[str, int]) -> dict[str, int]:
    if not windows:
        raise CorrelationFeatureError("At least one correlation window is required")

    validated_windows: dict[str, int] = {}

    for window_name, window_size in windows.items():
        if not isinstance(window_name, str) or not window_name.strip():
            raise CorrelationFeatureError("Correlation window names must be non-empty")

        if window_size <= 1:
            raise CorrelationFeatureError(
                "Correlation window sizes must be greater than 1"
            )

        validated_windows[window_name] = window_size

    return validated_windows


def _validate_pairs(pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    if not pairs:
        raise CorrelationFeatureError("At least one ticker pair is required")

    clean_pairs: list[tuple[str, str]] = []

    for ticker_a, ticker_b in pairs:
        clean_ticker_a = ticker_a.upper().strip()
        clean_ticker_b = ticker_b.upper().strip()

        if not clean_ticker_a or not clean_ticker_b:
            raise CorrelationFeatureError("Ticker pairs must be non-empty")

        if clean_ticker_a == clean_ticker_b:
            raise CorrelationFeatureError("Ticker pairs must contain different tickers")

        clean_pairs.append((clean_ticker_a, clean_ticker_b))

    return clean_pairs


def _validate_pair_tickers_exist(
    wide_returns: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
) -> None:
    available_tickers = set(wide_returns.columns)

    for ticker_a, ticker_b in pairs:
        missing_tickers = sorted({ticker_a, ticker_b}.difference(available_tickers))

        if missing_tickers:
            missing = ", ".join(missing_tickers)
            raise CorrelationFeatureError(
                f"Ticker pair contains unknown ticker(s): {missing}"
            )


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

    return wide_returns.sort_index()


def _average_off_diagonal_correlation(correlation_matrix: pd.DataFrame) -> float:
    values = correlation_matrix.to_numpy(dtype=float)
    mask = ~np.eye(values.shape[0], dtype=bool)
    off_diagonal_values = values[mask]

    return float(np.nanmean(off_diagonal_values))
