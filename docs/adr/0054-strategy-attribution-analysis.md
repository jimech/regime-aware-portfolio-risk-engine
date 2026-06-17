# ADR 0054: Strategy attribution analysis

## Status

Accepted

## Context

The project can evaluate whether a dynamic regime-aware strategy outperforms a static benchmark.

However, portfolio research should also explain why the strategy won or lost.

A strategy can outperform because of asset allocation, regime timing, defensive positioning, or real-asset exposure. It can underperform because of poor overweight decisions, missed rallies, or unnecessary de-risking.

## Decision

Add a strategy attribution module.

The first attribution layer decomposes dynamic-versus-static performance into:

- Static asset return contribution
- Dynamic asset return contribution
- Active asset contribution
- Average static weights
- Average dynamic weights
- Average active weights
- Regime-level active contribution, when regime labels are available

## Consequences

The project can now explain the source of dynamic strategy performance.

This improves the investment research quality of the system and supports more credible portfolio review narratives.

Future work can add Brinson attribution, factor attribution, and transaction-cost attribution.