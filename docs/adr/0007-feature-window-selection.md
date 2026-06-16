# ADR 0007: Feature window selection

## Status

Accepted

## Context

The regime-aware portfolio risk engine needs rolling financial features that capture short-term, medium-term, long-term, and annual market behavior.

Rolling returns and volatility are foundational inputs for regime detection because they help identify changing market environments such as low-volatility bull markets, high-volatility drawdowns, and recovery periods.

## Decision

Use the following default rolling windows:

```text
short: 21 trading days
medium: 63 trading days
long: 126 trading days
annual: 252 trading days