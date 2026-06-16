# ADR 0019: Transaction cost assumptions

## Status

Accepted

## Context

The dynamic regime-aware allocation strategy may trade more frequently than the static benchmark.

Backtests that ignore turnover and transaction costs can overstate performance.

The project needs a simple and transparent model for turnover and transaction costs.

## Decision

Implement one-way turnover and basis-point transaction costs.

One-way turnover is calculated as:

```text
0.5 * sum(abs(target_weight_i - current_weight_i))