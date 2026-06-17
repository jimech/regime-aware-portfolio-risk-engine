# ADR 0055: Factor exposure analysis

## Status

Accepted

## Context

Strategy returns alone do not explain the underlying risk exposures of a portfolio.

A dynamic strategy may outperform because it changes exposure to equity, defensive, real-asset, credit, or other risk factors.

Portfolio research should estimate whether a strategy is driven by factor exposures or by residual alpha.

## Decision

Add a factor exposure analysis module.

The module estimates linear factor exposures for each strategy return series using ordinary least squares.

It reports:

- Alpha
- Annualized alpha
- Factor betas
- R-squared
- Residual volatility
- Dominant factor by strategy
- Regime-conditioned factor exposures when regime labels are supplied

## Consequences

The project can now describe what risks explain static and dynamic strategy returns.

This makes the research output more useful for portfolio diagnostics, risk review, and investment interpretation.

Future work can add external factor libraries, rolling factor exposures, and statistical significance tests.