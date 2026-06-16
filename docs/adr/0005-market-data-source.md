# ADR 0005: Market data source

## Status

Accepted

## Context

The regime-aware portfolio risk engine needs historical price data for a multi-asset ETF universe.

The first version should prioritize simplicity, reproducibility, and ease of local development.

## Decision

Use `yfinance` as the initial data source for historical ETF prices.

The downloader will request automatically adjusted prices and store the resulting adjusted close values in a normalized long-format schema:

```text
date, ticker, adj_close

## Data validation approach

Because the first implementation uses an external research-oriented data source, the project validates downloaded price data before using it in downstream calculations.

The validation layer checks required schema fields, duplicate observations, missing prices, non-positive prices, expected ticker coverage, and incomplete observed date coverage.

The project distinguishes between errors and warnings.

Errors should block downstream processing because they can directly corrupt returns and risk metrics.

Warnings should be reviewed because they may reflect real market data limitations, ETF inception differences, trading calendar mismatches, or temporary data gaps.