# ADR 0072: Rolling factor exposure analysis

## Status

Accepted

## Context

The project supports factor exposure analysis, but a single full-period exposure estimate may hide important changes over time.

For a regime-aware research engine, rolling factor exposure is important because the strategy should be able to demonstrate how exposures change across market environments.

Examples include:

* Lower equity beta during stress regimes
* Higher defensive exposure during drawdowns
* Higher real-asset exposure during inflation-sensitive regimes
* More stable factor exposure after optimization

## Decision

Add a rolling factor exposure analysis module.

The module estimates rolling ordinary least squares factor betas using strategy returns and factor return inputs.

The output includes:

* Rolling alpha
* Rolling factor betas
* Rolling R-squared
* Observation count
* Residual volatility
* Dominant factor by absolute beta
* Factor-level summary statistics

## Consequences

The engine can now evaluate whether dynamic allocation changes its risk exposures over time.

This strengthens the research thesis by connecting regime-aware allocation decisions to observable changes in factor risk.

Future work can integrate rolling exposure tables directly into advanced research exports and memos.
