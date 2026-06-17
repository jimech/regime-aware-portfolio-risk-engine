# ADR 0051: Optimized market research workflow

## Status

Accepted

## Context

The market research workflow can run a regime-aware dynamic strategy, but earlier versions require manually specified regime weights.

The regime-aware optimizer can learn constrained portfolio weights from regime-conditioned asset behavior.

These capabilities should be connected so the project can produce an optimized dynamic strategy from price data.

## Decision

Add an optimized market research workflow.

The workflow:

1. Runs an initial market workflow with neutral equal-weight regime policies.
2. Uses the detected regimes and aligned asset returns to optimize one portfolio per regime.
3. Converts the optimized weight table into a regime allocation policy.
4. Re-runs the market workflow using the optimizer-learned regime policy.

## Consequences

The project no longer depends only on hard-coded dynamic allocation weights.

It can estimate regime-specific portfolio weights from historical market behavior.

This makes the workflow more realistic as a quantitative portfolio research engine.

Future work can replace the initial in-sample optimization with walk-forward out-of-sample optimization.