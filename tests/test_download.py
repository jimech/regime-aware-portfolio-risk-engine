from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from regime_risk_engine.data.download import (
    PRICE_COLUMNS,
    PriceDownloadError,
    download_adjusted_prices,
    normalize_yfinance_prices,
)
from regime_risk_engine.data.storage import load_prices_from_csv, save_prices_to_csv


def test_download_adjusted_prices_requires_tickers() -> None:
    with pytest.raises(PriceDownloadError, match="At least one ticker"):
        download_adjusted_prices([], "2020-01-01")


def test_download_adjusted_prices_raises_for_empty_data() -> None:
    def fake_download(**_: Any) -> pd.DataFrame:
        return pd.DataFrame()

    with pytest.raises(PriceDownloadError, match="empty"):
        download_adjusted_prices(["SPY"], "2020-01-01", fake_download)


def test_normalize_single_ticker_prices() -> None:
    raw_prices = pd.DataFrame(
        {
            "Close": [100.0, 101.5],
            "Open": [99.0, 100.5],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )

    prices = normalize_yfinance_prices(raw_prices, ["SPY"])

    assert list(prices.columns) == PRICE_COLUMNS
    assert prices["ticker"].tolist() == ["SPY", "SPY"]
    assert prices["adj_close"].tolist() == [100.0, 101.5]


def test_normalize_multi_ticker_prices_grouped_by_ticker() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("SPY", "Close"),
            ("SPY", "Open"),
            ("TLT", "Close"),
            ("TLT", "Open"),
        ]
    )
    raw_prices = pd.DataFrame(
        [
            [100.0, 99.0, 120.0, 119.0],
            [101.0, 100.0, 121.0, 120.0],
        ],
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        columns=columns,
    )

    prices = normalize_yfinance_prices(raw_prices, ["SPY", "TLT"])

    assert list(prices.columns) == PRICE_COLUMNS
    assert len(prices) == 4
    assert set(prices["ticker"]) == {"SPY", "TLT"}


def test_download_adjusted_prices_uses_injected_downloader() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("SPY", "Close"),
            ("TLT", "Close"),
        ]
    )
    raw_prices = pd.DataFrame(
        [
            [100.0, 120.0],
            [101.0, 121.0],
        ],
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        columns=columns,
    )

    def fake_download(**_: Any) -> pd.DataFrame:
        return raw_prices

    prices = download_adjusted_prices(["SPY", "TLT"], "2020-01-01", fake_download)

    assert len(prices) == 4
    assert set(prices["ticker"]) == {"SPY", "TLT"}


def test_save_and_load_prices_to_csv(tmp_path: Path) -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "ticker": ["SPY", "SPY"],
            "adj_close": [100.0, 101.0],
        }
    )
    output_path = tmp_path / "prices.csv"

    saved_path = save_prices_to_csv(prices, output_path)
    loaded_prices = load_prices_from_csv(saved_path)

    assert saved_path == output_path
    pd.testing.assert_frame_equal(loaded_prices, prices)
