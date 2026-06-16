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