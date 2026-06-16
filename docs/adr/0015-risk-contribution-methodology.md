# ADR 0015: Risk contribution methodology

## Status

Accepted

## Context

Portfolio risk is driven by asset weights, individual asset volatility, and covariance between assets.

A portfolio can have a large allocation to an asset that contributes little risk, or a small allocation to an asset that contributes significant risk.

The project needs risk contribution analytics to explain which assets drive portfolio volatility within each regime.

## Decision

Implement portfolio risk contribution analytics using covariance matrices and portfolio weights.

The first implementation calculates:

- Portfolio volatility
- Marginal risk contribution
- Component risk contribution
- Percentage risk contribution

Component risk contribution is calculated as:

```text
weight_i * marginal_risk_contribution_i