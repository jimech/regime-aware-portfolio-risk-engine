from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CrisisWindow:
    """Named historical market stress window."""

    name: str
    start_date: str
    end_date: str
    description: str
    category: str

    def to_record(self) -> dict[str, str]:
        """Return a CSV-friendly crisis window record."""
        return {
            "name": self.name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "description": self.description,
            "category": self.category,
        }


DEFAULT_CRISIS_WINDOWS: tuple[CrisisWindow, ...] = (
    CrisisWindow(
        name="global_financial_crisis",
        start_date="2007-10-09",
        end_date="2009-03-09",
        description=(
            "Equity bear market and systemic credit crisis around the "
            "2008 Global Financial Crisis."
        ),
        category="credit_crisis",
    ),
    CrisisWindow(
        name="eurozone_debt_crisis",
        start_date="2011-07-22",
        end_date="2011-10-03",
        description=(
            "Risk-off period associated with eurozone sovereign stress "
            "and the United States debt-ceiling downgrade period."
        ),
        category="sovereign_stress",
    ),
    CrisisWindow(
        name="volmageddon",
        start_date="2018-01-26",
        end_date="2018-02-08",
        description=(
            "Volatility spike associated with the February 2018 unwind "
            "of short-volatility products."
        ),
        category="volatility_shock",
    ),
    CrisisWindow(
        name="q4_2018_selloff",
        start_date="2018-09-20",
        end_date="2018-12-24",
        description=(
            "Equity selloff associated with tightening financial conditions, "
            "growth concerns, and Federal Reserve policy uncertainty."
        ),
        category="growth_scare",
    ),
    CrisisWindow(
        name="covid_crash",
        start_date="2020-02-19",
        end_date="2020-03-23",
        description=(
            "Rapid global equity drawdown during the initial COVID-19 pandemic shock."
        ),
        category="pandemic_shock",
    ),
    CrisisWindow(
        name="inflation_rate_shock_2022",
        start_date="2022-01-03",
        end_date="2022-10-12",
        description=(
            "Inflation and interest-rate shock period during the 2022 "
            "equity and bond market drawdown."
        ),
        category="inflation_rate_shock",
    ),
    CrisisWindow(
        name="regional_bank_stress_2023",
        start_date="2023-03-08",
        end_date="2023-05-04",
        description=(
            "Regional banking stress period following several United States "
            "bank failures and deposit-flight concerns."
        ),
        category="banking_stress",
    ),
)


def list_crisis_windows() -> pd.DataFrame:
    """Return the default historical crisis windows as a DataFrame."""
    return pd.DataFrame(
        [window.to_record() for window in DEFAULT_CRISIS_WINDOWS],
        columns=["name", "start_date", "end_date", "description", "category"],
    )


def get_crisis_window(name: str) -> CrisisWindow:
    """Return a crisis window by name.

    Parameters
    ----------
    name:
        Crisis window name, case-insensitive.

    Raises
    ------
    ValueError
        If no crisis window with the requested name exists.
    """
    normalized_name = name.strip().lower()

    for window in DEFAULT_CRISIS_WINDOWS:
        if window.name.lower() == normalized_name:
            return window

    available = ", ".join(window.name for window in DEFAULT_CRISIS_WINDOWS)
    raise ValueError(f"Unknown crisis window '{name}'. Available windows: {available}")


def crisis_windows_to_stress_periods(
    windows: tuple[CrisisWindow, ...] | None = None,
) -> pd.DataFrame:
    """Return crisis windows in stress-testing input format."""
    selected_windows = DEFAULT_CRISIS_WINDOWS if windows is None else windows

    return pd.DataFrame(
        [
            {
                "period": window.name,
                "start_date": window.start_date,
                "end_date": window.end_date,
            }
            for window in selected_windows
        ],
        columns=["period", "start_date", "end_date"],
    )


def filter_crisis_windows_for_date_range(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    windows: tuple[CrisisWindow, ...] | None = None,
) -> tuple[CrisisWindow, ...]:
    """Return crisis windows that overlap a date range."""
    selected_windows = DEFAULT_CRISIS_WINDOWS if windows is None else windows

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")

    overlapping_windows: list[CrisisWindow] = []

    for window in selected_windows:
        window_start = pd.Timestamp(window.start_date)
        window_end = pd.Timestamp(window.end_date)

        if window_start <= end and window_end >= start:
            overlapping_windows.append(window)

    return tuple(overlapping_windows)


def filter_crisis_windows_for_index(
    index: pd.Index,
    windows: tuple[CrisisWindow, ...] | None = None,
) -> tuple[CrisisWindow, ...]:
    """Return crisis windows that overlap a pandas date index."""
    if index.empty:
        return tuple()

    date_index = pd.DatetimeIndex(index)
    return filter_crisis_windows_for_date_range(
        start_date=date_index.min(),
        end_date=date_index.max(),
        windows=windows,
    )


def write_crisis_windows_csv(
    output_path: str | Path,
    windows: tuple[CrisisWindow, ...] | None = None,
) -> Path:
    """Write crisis windows to CSV and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    selected_windows = DEFAULT_CRISIS_WINDOWS if windows is None else windows
    frame = pd.DataFrame(
        [window.to_record() for window in selected_windows],
        columns=["name", "start_date", "end_date", "description", "category"],
    )
    frame.to_csv(path, index=False)

    return path
