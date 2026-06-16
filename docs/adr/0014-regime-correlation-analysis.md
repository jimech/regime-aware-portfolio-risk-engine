# ADR 0014: Regime correlation analysis

## Status

Accepted

## Context

Portfolio risk depends on both individual asset risk and the relationships between assets.

A portfolio may appear diversified during normal markets but become less diversified during stress regimes if asset correlations rise.

The project needs regime-specific correlation and covariance analytics to evaluate how diversification changes across market environments.

## Decision

Add regime-specific correlation and covariance analytics.

For each detected regime, calculate:

- Asset correlation matrix
- Annualized asset covariance matrix
- Average pairwise correlation summary

Returns and regime labels are aligned by date before analytics are calculated.

## Consequences

The project can analyze whether diversification improves or deteriorates across regimes.

Correlation matrices help explain changing diversification behavior.

Covariance matrices prepare the project for later portfolio risk contribution and allocation work.

The results depend on the number of observations in each regime. Regimes with very few observations should be interpreted carefully.