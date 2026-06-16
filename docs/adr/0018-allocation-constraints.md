# ADR 0018: Allocation constraints

## Status

Accepted

## Context

A dynamic regime-aware allocation strategy can become unrealistic if it is allowed to move freely between extreme portfolios.

Realistic portfolio construction usually requires constraints such as asset-level limits, asset-class exposure limits, and turnover controls.

The project needs a constraint layer before dynamic allocation is backtested.

## Decision

Implement allocation constraint validation.

The first constraint layer supports:

- Asset-level minimum and maximum weights
- Asset-class minimum and maximum exposures
- Maximum turnover validation

Portfolio weights must be non-negative and sum to 1.0.

Asset-class exposure is calculated by summing weights across tickers mapped to the same asset class.

Turnover is calculated as one-way turnover:

```text
0.5 * sum(abs(target_weight_i - current_weight_i))