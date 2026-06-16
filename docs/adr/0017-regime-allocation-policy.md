# ADR 0017: Regime allocation policy

## Status

Accepted

## Context

The project compares a static portfolio against a dynamic regime-aware allocation strategy.

Regime detection models produce numeric regime labels, but portfolio decisions require target asset weights.

The project needs a policy layer that maps detected regimes to target allocations.

## Decision

Implement a rule-based regime allocation policy.

Each numeric regime maps to a full set of target portfolio weights.

Weights must:

- Use known tickers
- Be non-negative
- Sum to 1.0

The policy also supports an optional fallback allocation for unknown regimes.

## Consequences

The project now has a clear bridge between regime detection and dynamic allocation.

This makes the strategy transparent and easy to test.

The first policy layer is rule-based rather than optimized. Later tickets may add constraints, turnover limits, transaction costs, and optimization-based allocation.