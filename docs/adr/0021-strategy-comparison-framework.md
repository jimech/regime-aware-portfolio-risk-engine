# ADR 0021: Strategy comparison framework

## Status

Accepted

## Context

The project compares a static benchmark portfolio against a dynamic regime-aware allocation strategy.

Both strategies should be evaluated using the same backtest engine, transaction-cost assumptions, date alignment, and risk metrics.

## Decision

Implement a strategy comparison layer.

The comparison layer runs:

- Static strategy backtest
- Dynamic strategy backtest
- Net return alignment
- Risk metric summary
- Candidate-minus-benchmark metric deltas

The first comparison treats the static strategy as the benchmark and the dynamic strategy as the candidate.

Metrics are calculated on overlapping net return dates to ensure a fair comparison.

## Consequences

The project can directly evaluate whether the dynamic regime-aware strategy improves performance, risk, drawdown, or tail risk relative to the static benchmark.

The comparison framework is reusable for future strategies beyond the initial static and dynamic pair.