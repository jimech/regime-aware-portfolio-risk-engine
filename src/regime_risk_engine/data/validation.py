from dataclasses import dataclass
from typing import Literal

import pandas as pd

REQUIRED_PRICE_COLUMNS = {"date", "ticker", "adj_close"}

Severity = Literal["warning", "error"]


@dataclass(frozen=True, slots=True)
class DataValidationIssue:
    """Structured data validation issue."""

    severity: Severity
    check_name: str
    message: str
    ticker: str | None = None


class DataValidationError(ValueError):
    """Raised when data validation finds blocking errors."""


def validate_price_data(
    prices: pd.DataFrame,
    expected_tickers: list[str] | None = None,
) -> list[DataValidationIssue]:
    """Validate normalized historical price data.

    Args:
        prices: Long-format price data with date, ticker, and adj_close columns.
        expected_tickers: Optional list of tickers expected to appear in the data.

    Returns:
        Structured validation issues.
    """
    issues: list[DataValidationIssue] = []

    issues.extend(_validate_required_columns(prices))

    if issues:
        return issues

    if prices.empty:
        return [
            DataValidationIssue(
                severity="error",
                check_name="empty_data",
                message="Price data is empty",
            )
        ]

    normalized_prices = prices.copy()
    normalized_prices["date"] = pd.to_datetime(
        normalized_prices["date"],
        errors="coerce",
    )
    normalized_prices["ticker"] = (
        normalized_prices["ticker"].astype(str).str.upper().str.strip()
    )

    issues.extend(_validate_dates(normalized_prices))
    issues.extend(_validate_duplicate_rows(normalized_prices))
    issues.extend(_validate_missing_prices(normalized_prices))
    issues.extend(_validate_non_positive_prices(normalized_prices))

    if expected_tickers is not None:
        issues.extend(_validate_expected_tickers(normalized_prices, expected_tickers))
        issues.extend(_validate_missing_observed_dates(normalized_prices))

    return issues


def raise_if_price_data_invalid(
    prices: pd.DataFrame,
    expected_tickers: list[str] | None = None,
) -> None:
    """Raise an error if price data has validation errors."""
    issues = validate_price_data(prices, expected_tickers=expected_tickers)
    errors = [issue for issue in issues if issue.severity == "error"]

    if errors:
        messages = "; ".join(issue.message for issue in errors)
        raise DataValidationError(messages)


def has_errors(issues: list[DataValidationIssue]) -> bool:
    """Return True if validation issues contain at least one error."""
    return any(issue.severity == "error" for issue in issues)


def _validate_required_columns(prices: pd.DataFrame) -> list[DataValidationIssue]:
    missing_columns = REQUIRED_PRICE_COLUMNS.difference(prices.columns)

    if not missing_columns:
        return []

    missing = ", ".join(sorted(missing_columns))

    return [
        DataValidationIssue(
            severity="error",
            check_name="required_columns",
            message=f"Missing required price column(s): {missing}",
        )
    ]


def _validate_dates(prices: pd.DataFrame) -> list[DataValidationIssue]:
    invalid_dates = prices["date"].isna()

    if not invalid_dates.any():
        return []

    return [
        DataValidationIssue(
            severity="error",
            check_name="invalid_dates",
            message=f"Price data contains {int(invalid_dates.sum())} invalid date(s)",
        )
    ]


def _validate_duplicate_rows(prices: pd.DataFrame) -> list[DataValidationIssue]:
    duplicate_rows = prices.duplicated(subset=["date", "ticker"])

    if not duplicate_rows.any():
        return []

    return [
        DataValidationIssue(
            severity="error",
            check_name="duplicate_rows",
            message=(
                "Price data contains "
                f"{int(duplicate_rows.sum())} duplicate date/ticker row(s)"
            ),
        )
    ]


def _validate_missing_prices(prices: pd.DataFrame) -> list[DataValidationIssue]:
    missing_prices = prices["adj_close"].isna()

    if not missing_prices.any():
        return []

    issues: list[DataValidationIssue] = []

    for ticker, _ticker_prices in prices[missing_prices].groupby("ticker"):
        issues.append(
            DataValidationIssue(
                severity="error",
                check_name="missing_prices",
                message=f"Ticker {ticker} contains missing adjusted close value(s)",
                ticker=str(ticker),
            )
        )

    return issues


def _validate_non_positive_prices(prices: pd.DataFrame) -> list[DataValidationIssue]:
    numeric_prices = pd.to_numeric(prices["adj_close"], errors="coerce")
    non_positive_prices = numeric_prices <= 0

    if not non_positive_prices.any():
        return []

    issues: list[DataValidationIssue] = []

    for ticker, _ticker_prices in prices[non_positive_prices].groupby("ticker"):
        issues.append(
            DataValidationIssue(
                severity="error",
                check_name="non_positive_prices",
                message=(
                    f"Ticker {ticker} contains non-positive adjusted close value(s)"
                ),
                ticker=str(ticker),
            )
        )

    return issues


def _validate_expected_tickers(
    prices: pd.DataFrame,
    expected_tickers: list[str],
) -> list[DataValidationIssue]:
    expected = {ticker.upper().strip() for ticker in expected_tickers}
    observed = set(prices["ticker"].dropna().unique())

    missing_tickers = sorted(expected.difference(observed))
    unexpected_tickers = sorted(observed.difference(expected))

    issues: list[DataValidationIssue] = []

    for ticker in missing_tickers:
        issues.append(
            DataValidationIssue(
                severity="error",
                check_name="missing_expected_ticker",
                message=f"Expected ticker {ticker} is missing from price data",
                ticker=ticker,
            )
        )

    for ticker in unexpected_tickers:
        issues.append(
            DataValidationIssue(
                severity="warning",
                check_name="unexpected_ticker",
                message=f"Unexpected ticker {ticker} appears in price data",
                ticker=ticker,
            )
        )

    return issues


def _validate_missing_observed_dates(
    prices: pd.DataFrame,
) -> list[DataValidationIssue]:
    """Check whether each ticker has all dates observed in the full dataset.

    This avoids comparing against a calendar-day or business-day range, which would
    incorrectly flag weekends and market holidays as missing.
    """
    all_observed_dates = set(prices["date"].dropna().unique())

    issues: list[DataValidationIssue] = []

    for ticker, ticker_prices in prices.groupby("ticker"):
        ticker_dates = set(ticker_prices["date"].dropna().unique())
        missing_dates = sorted(all_observed_dates.difference(ticker_dates))

        if missing_dates:
            issues.append(
                DataValidationIssue(
                    severity="warning",
                    check_name="missing_observed_dates",
                    message=(
                        f"Ticker {ticker} is missing "
                        f"{len(missing_dates)} observed date(s)"
                    ),
                    ticker=str(ticker),
                )
            )

    return issues
