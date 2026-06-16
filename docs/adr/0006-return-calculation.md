# ADR 0006: Return calculation

## Status

Accepted

## Context

The regime-aware portfolio risk engine needs processed return data for feature engineering, regime detection, risk analytics, allocation, and backtesting.

Raw adjusted price data must be transformed into a consistent return dataset.

## Decision

Use adjusted close prices as the default input for return calculations.

The project will support both simple returns and log returns.

Simple returns are calculated as:

```text
price_t / price_t-1 - 1