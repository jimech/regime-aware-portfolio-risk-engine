# ADR 0053: Stress-period strategy analysis

## Status

Accepted

## Context

Overall backtest metrics are useful, but portfolio research also needs to evaluate how a strategy behaves during market stress.

A regime-aware strategy is especially valuable if it can reduce drawdowns, protect capital, or improve risk-adjusted returns during difficult market environments.

## Decision

Add a stress-period analysis module.

The module compares a candidate strategy against a benchmark across named stress windows.

For each stress period, it calculates:

- Cumulative return
- Max drawdown
- Annualized volatility
- Sharpe ratio
- Return delta
- Drawdown delta
- Volatility delta
- Dominant regime, when regime labels are available
- Capital-protection assessment

## Consequences

The project can now evaluate whether a regime-aware strategy adds value during market stress.

This makes the research output more useful for portfolio review and risk-management discussions.

Future work can add predefined historical crisis windows such as the global financial crisis, COVID crash, inflation shock, and rate-hiking cycles.