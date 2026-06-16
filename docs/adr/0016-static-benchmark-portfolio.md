# ADR 0016: Static benchmark portfolio

## Status

Accepted

## Context

The project compares a static portfolio against a dynamic regime-aware allocation strategy.

A static benchmark is needed so the project can evaluate whether regime-aware allocation improves risk-adjusted performance, drawdowns, or tail risk.

## Decision

Implement a static benchmark allocation module.

The static benchmark uses fixed portfolio weights across the full backtest period.

Weights must:

- Map to available tickers
- Be non-negative
- Sum to 1.0

Static portfolio returns are calculated from long-format asset returns by pivoting to a date-by-ticker matrix and applying the fixed weight vector.

## Consequences

The project now has a clear baseline strategy.

The static benchmark is simple and transparent.

Future dynamic strategies will be compared against this baseline using the same backtesting and risk metric framework.