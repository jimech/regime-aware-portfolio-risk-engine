from pathlib import Path

import pandas as pd


def save_prices_to_csv(prices: pd.DataFrame, output_path: str | Path) -> Path:
    """Save normalized price data to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    prices.to_csv(path, index=False)

    return path


def load_prices_from_csv(input_path: str | Path) -> pd.DataFrame:
    """Load normalized price data from CSV."""
    path = Path(input_path)

    return pd.read_csv(path, parse_dates=["date"])
