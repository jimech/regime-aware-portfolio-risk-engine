# ADR 0052: Walk-forward regime optimization

## Status

Accepted

## Context

The regime-aware optimizer can estimate optimized portfolios from regime-conditioned returns.

However, optimizing on the full sample can overstate strategy quality because future information may influence portfolio weights.

A more credible research design should train allocation rules on past data and apply them to future data.

## Decision

Add walk-forward regime optimization.

The walk-forward optimizer:

1. Uses a rolling training window.
2. Optimizes regime-specific portfolio weights using only training data.
3. Applies the learned policy to the next test window.
4. Repeats the process through time.
5. Compares walk-forward dynamic returns against a static benchmark.

If a training window does not contain enough regime diversity, the workflow falls back to benchmark weights.

## Consequences

The project now supports a more realistic out-of-sample allocation test.

This makes the optimizer more credible as a quantitative research tool.

Future work can extend this to retrain the regime detection model itself in each walk-forward window.