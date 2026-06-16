from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

import pandas as pd

PriceDownloadFunction = Callable[..., pd.DataFrame]

PRICE_COLUMNS = ["date", "ticker", "adj_close"]


class PriceDownloadError(RuntimeError):
    """Raised when historical price data cannot be downloaded or parsed."""


def download_adjusted_prices(
    tickers: Sequence[str],
    start_date: str | date,
    end_date: str | date | PriceDownloadFunction | None = None,
    download_function: PriceDownloadFunction | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers.

    Args:
        tickers: Asset tickers to download.
        start_date: Inclusive start date.
        end_date: Optional exclusive end date.
        download_function: Optional injectable downloader for tests.

    Returns:
        Long-format DataFrame with date, ticker, and adj_close columns.

    Raises:
        PriceDownloadError: If no tickers are provided or no data is returned.
    """
    clean_tickers = [ticker.upper().strip() for ticker in tickers if ticker.strip()]

    if not clean_tickers:
        raise PriceDownloadError("At least one ticker is required")

    if callable(end_date) and download_function is None:
        download_function = end_date
        end_date = None

    if download_function is None:
        import yfinance as yf

        download_function = yf.download

    raw_prices = download_function(
        tickers=clean_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw_prices.empty:
        raise PriceDownloadError("Downloaded price data is empty")

    prices = normalize_yfinance_prices(raw_prices, clean_tickers)

    if prices.empty:
        raise PriceDownloadError("No adjusted close prices could be parsed")

    return prices


def normalize_yfinance_prices(
    raw_prices: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    """Normalize yfinance output into long-format adjusted close prices."""
    if isinstance(raw_prices.columns, pd.MultiIndex):
        return _normalize_multi_index_prices(raw_prices, tickers)

    return _normalize_single_ticker_prices(raw_prices, tickers)


def _normalize_multi_index_prices(
    raw_prices: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    price_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        close_series = _extract_close_series(raw_prices, ticker)

        if close_series is None:
            continue

        price_frame = close_series.rename("adj_close").reset_index()
        price_frame["ticker"] = ticker
        price_frame = price_frame.rename(columns={price_frame.columns[0]: "date"})
        price_frames.append(price_frame[PRICE_COLUMNS])

    if not price_frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    return pd.concat(price_frames, ignore_index=True).dropna(subset=["adj_close"])


def _normalize_single_ticker_prices(
    raw_prices: pd.DataFrame,
    tickers: Sequence[str],
) -> pd.DataFrame:
    if "Close" not in raw_prices.columns:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    if len(tickers) != 1:
        raise PriceDownloadError("Flat price data received for multiple tickers")

    ticker = tickers[0]
    price_frame = raw_prices["Close"].rename("adj_close").reset_index()
    price_frame["ticker"] = ticker
    price_frame = price_frame.rename(columns={price_frame.columns[0]: "date"})

    return price_frame[PRICE_COLUMNS].dropna(subset=["adj_close"])


def _extract_close_series(
    raw_prices: pd.DataFrame,
    ticker: str,
) -> pd.Series | None:
    columns = raw_prices.columns

    if (ticker, "Close") in columns:
        return _to_numeric_series(raw_prices[(ticker, "Close")])

    if ("Close", ticker) in columns:
        return _to_numeric_series(raw_prices[("Close", ticker)])

    return None


def _to_numeric_series(series: Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")
