# ADR 0046: Market research workflow

## Status

Accepted

## Context

The project has separate modules for returns, regimes, backtesting, regime evaluation, reporting, and investment research summaries.

A practical research workflow needs to connect these capabilities from real price data to final investment conclusions.

## Decision

Add a market research workflow in the research layer.

The first workflow starts from a complete long-format price panel and performs:

- Simple return calculation
- Rolling feature construction
- K-Means regime detection
- Static portfolio backtest
- Regime-aware dynamic portfolio backtest
- Strategy metric comparison
- Regime-conditioned evaluation
- Investment research summary generation

Dynamic regime-aware weights are lagged by one period before being applied to returns to reduce look-ahead bias.

## Consequences

The project can now produce a complete research result from market price data.

The workflow is intentionally explicit and interpretable.

Future work can replace the simple K-Means feature set with the richer feature and validation modules already available in the project.