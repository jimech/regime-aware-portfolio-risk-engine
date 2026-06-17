# ADR 0049: Regime intelligence and market-state labeling

## Status

Accepted

## Context

The project can detect regimes, but numeric regime identifiers are not directly useful for investment interpretation.

A portfolio manager needs to understand what each regime represents economically.

Examples include:

- Growth / risk-on
- Defensive / stress
- Inflation / real assets
- Low-volatility grind
- Mixed / transition

## Decision

Add a regime intelligence layer.

The layer profiles each detected regime using:

- Returns
- Volatility
- Sharpe ratio
- Max drawdown
- Average correlation
- Asset leadership
- Equity behavior
- Defensive asset behavior
- Real asset behavior

The layer assigns interpretable market-state labels using rule-based financial diagnostics.

## Consequences

The project can now explain regimes in economic terms instead of only reporting cluster IDs.

This makes the research output more useful for portfolio review, strategy diagnostics, and investment memo generation.

Future work can integrate these labels directly into the memo and allocation optimizer.