# ADR 0057: Regime scenario simulation

## Status

Accepted

## Context

Historical backtests explain what happened in the observed sample, but portfolio risk management also needs forward-looking scenario analysis.

Detected regimes and transition probabilities can be used to simulate possible future regime paths.

Those simulated paths can help estimate the distribution of future static and dynamic strategy outcomes.

## Decision

Add a regime scenario simulation module.

The module:

- Estimates regime transition probabilities from historical labels
- Simulates future regime paths
- Samples historical strategy returns conditional on simulated regimes
- Tracks simulated strategy wealth paths
- Summarizes terminal return distributions
- Reports probability of loss, VaR, CVaR, best case, and worst case

## Consequences

The project now supports forward-looking regime-conditioned scenario analysis.

This helps connect regime detection to practical portfolio risk management.

Future work can add parametric return models, correlated factor simulations, and scenario-specific macro assumptions.