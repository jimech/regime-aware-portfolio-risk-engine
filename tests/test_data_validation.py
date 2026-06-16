import pandas as pd
import pytest

from regime_risk_engine.data.validation import (
    DataValidationError,
    DataValidationIssue,
    has_errors,
    raise_if_price_data_invalid,
    validate_price_data,
)


def make_valid_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-01",
                    "2020-01-02",
                ]
            ),
            "ticker": ["SPY", "SPY", "TLT", "TLT"],
            "adj_close": [100.0, 101.0, 120.0, 121.0],
        }
    )


def test_validate_price_data_returns_no_issues_for_valid_data() -> None:
    prices = make_valid_prices()

    issues = validate_price_data(prices, expected_tickers=["SPY", "TLT"])

    assert issues == []


def test_validate_price_data_detects_missing_required_columns() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
        }
    )

    issues = validate_price_data(prices)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].check_name == "required_columns"
    assert "adj_close" in issues[0].message


def test_validate_price_data_detects_empty_data() -> None:
    prices = pd.DataFrame(columns=["date", "ticker", "adj_close"])

    issues = validate_price_data(prices)

    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].check_name == "empty_data"


def test_validate_price_data_detects_invalid_dates() -> None:
    prices = pd.DataFrame(
        {
            "date": ["not-a-date"],
            "ticker": ["SPY"],
            "adj_close": [100.0],
        }
    )

    issues = validate_price_data(prices)

    assert has_errors(issues)
    assert any(issue.check_name == "invalid_dates" for issue in issues)


def test_validate_price_data_detects_duplicate_date_ticker_rows() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "ticker": ["SPY", "SPY"],
            "adj_close": [100.0, 101.0],
        }
    )

    issues = validate_price_data(prices)

    assert has_errors(issues)
    assert any(issue.check_name == "duplicate_rows" for issue in issues)


def test_validate_price_data_detects_missing_prices() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "ticker": ["SPY", "SPY"],
            "adj_close": [100.0, None],
        }
    )

    issues = validate_price_data(prices)

    assert has_errors(issues)
    assert any(issue.check_name == "missing_prices" for issue in issues)


def test_validate_price_data_detects_non_positive_prices() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "ticker": ["SPY", "SPY"],
            "adj_close": [100.0, 0.0],
        }
    )

    issues = validate_price_data(prices)

    assert has_errors(issues)
    assert any(issue.check_name == "non_positive_prices" for issue in issues)


def test_validate_price_data_detects_missing_expected_ticker() -> None:
    prices = make_valid_prices()

    issues = validate_price_data(prices, expected_tickers=["SPY", "TLT", "GLD"])

    assert has_errors(issues)
    assert any(issue.check_name == "missing_expected_ticker" for issue in issues)


def test_validate_price_data_detects_unexpected_ticker() -> None:
    prices = make_valid_prices()

    issues = validate_price_data(prices, expected_tickers=["SPY"])

    assert any(issue.severity == "warning" for issue in issues)
    assert any(issue.check_name == "unexpected_ticker" for issue in issues)


def test_validate_price_data_detects_missing_observed_dates() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-01",
                    "2020-01-03",
                ]
            ),
            "ticker": ["SPY", "SPY", "SPY", "TLT", "TLT"],
            "adj_close": [100.0, 101.0, 102.0, 120.0, 122.0],
        }
    )

    issues = validate_price_data(prices, expected_tickers=["SPY", "TLT"])

    assert any(issue.check_name == "missing_observed_dates" for issue in issues)
    assert any(issue.ticker == "TLT" for issue in issues)


def test_raise_if_price_data_invalid_raises_for_errors() -> None:
    prices = pd.DataFrame(columns=["date", "ticker", "adj_close"])

    with pytest.raises(DataValidationError, match="Price data is empty"):
        raise_if_price_data_invalid(prices)


def test_raise_if_price_data_invalid_does_not_raise_for_valid_data() -> None:
    prices = make_valid_prices()

    raise_if_price_data_invalid(prices, expected_tickers=["SPY", "TLT"])


def test_has_errors_returns_false_when_only_warnings_exist() -> None:
    issues = [
        DataValidationIssue(
            severity="warning",
            check_name="example_warning",
            message="Example warning",
        )
    ]

    assert has_errors(issues) is False
