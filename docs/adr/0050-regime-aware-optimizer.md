# ADR 0050: Regime-aware portfolio optimizer

## Status

Accepted

## Context

The market research workflow can compare static and dynamic strategies, but the first dynamic allocation design uses manually specified regime weights.

Manual regime weights make the strategy interpretable, but they are not advanced enough for a professional quant portfolio research engine.

The project needs a layer that can estimate risk-aware portfolio weights from regime-conditioned return behavior.

## Decision

Add a regime-aware portfolio optimizer.

The optimizer estimates one portfolio per detected regime using:

- Regime-conditioned asset returns
- Expected return
- Annualized volatility
- Historical CVaR
- Turnover from a benchmark portfolio
- Minimum and maximum weight constraints

The first optimizer uses deterministic candidate portfolios and seeded random candidate portfolios. This avoids adding a heavy optimization dependency while still producing constrained, risk-aware allocations.

## Consequences

The project can now learn allocation weights from regime behavior instead of relying only on hard-coded policy weights.

This moves the engine closer to realistic quantitative portfolio research.

Future work can integrate convex optimization, transaction-cost-aware rebalancing, and walk-forward out-of-sample optimization.