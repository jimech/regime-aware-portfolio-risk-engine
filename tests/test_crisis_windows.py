from pathlib import Path

import pandas as pd
import pytest

from regime_risk_engine.research.crisis_windows import (
    DEFAULT_CRISIS_WINDOWS,
    crisis_windows_to_stress_periods,
    filter_crisis_windows_for_date_range,
    filter_crisis_windows_for_index,
    get_crisis_window,
    list_crisis_windows,
    write_crisis_windows_csv,
)


def test_default_crisis_windows_include_major_market_stress_periods() -> None:
    names = {window.name for window in DEFAULT_CRISIS_WINDOWS}

    assert "global_financial_crisis" in names
    assert "covid_crash" in names
    assert "inflation_rate_shock_2022" in names
    assert "q4_2018_selloff" in names


def test_list_crisis_windows_returns_documented_frame() -> None:
    frame = list_crisis_windows()

    assert list(frame.columns) == [
        "name",
        "start_date",
        "end_date",
        "description",
        "category",
    ]
    assert "covid_crash" in set(frame["name"])
    assert frame["description"].str.len().min() > 0


def test_get_crisis_window_returns_named_window() -> None:
    window = get_crisis_window("covid_crash")

    assert window.name == "covid_crash"
    assert window.start_date == "2020-02-19"
    assert window.end_date == "2020-03-23"


def test_get_crisis_window_is_case_insensitive() -> None:
    window = get_crisis_window("COVID_CRASH")

    assert window.name == "covid_crash"


def test_get_crisis_window_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown crisis window"):
        get_crisis_window("not_a_real_crisis")


def test_crisis_windows_to_stress_periods_matches_stress_input_shape() -> None:
    stress_periods = crisis_windows_to_stress_periods()

    assert list(stress_periods.columns) == ["period", "start_date", "end_date"]
    assert "covid_crash" in set(stress_periods["period"])


def test_filter_crisis_windows_for_date_range_returns_overlapping_windows() -> None:
    windows = filter_crisis_windows_for_date_range(
        start_date="2020-01-01",
        end_date="2020-12-31",
    )

    names = {window.name for window in windows}

    assert names == {"covid_crash"}


def test_filter_crisis_windows_for_date_range_rejects_invalid_range() -> None:
    with pytest.raises(
        ValueError,
        match="end_date must be greater than or equal to start_date",
    ):
        filter_crisis_windows_for_date_range(
            start_date="2021-01-01",
            end_date="2020-01-01",
        )


def test_filter_crisis_windows_for_index_handles_empty_index() -> None:
    windows = filter_crisis_windows_for_index(pd.Index([]))

    assert windows == tuple()


def test_filter_crisis_windows_for_index_uses_index_bounds() -> None:
    index = pd.date_range("2022-01-01", "2022-12-31", freq="D")

    windows = filter_crisis_windows_for_index(index)
    names = {window.name for window in windows}

    assert "inflation_rate_shock_2022" in names


def test_write_crisis_windows_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "crisis_windows.csv"

    result_path = write_crisis_windows_csv(output_path)

    assert result_path == output_path
    assert output_path.exists()

    frame = pd.read_csv(output_path)

    assert "covid_crash" in set(frame["name"])
    assert "category" in frame.columns
