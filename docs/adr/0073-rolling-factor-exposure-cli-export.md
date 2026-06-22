# ADR 0073: Rolling factor exposure CLI export

## Status

Accepted

## Context

The project now supports rolling factor exposure analysis as a research module.

However, users should be able to run rolling exposure analysis from CSV inputs without writing Python code. This is consistent with the project’s existing CLI-first research workflow design.

## Decision

Add an `export-rolling-factor-exposure` CLI command.

The command reads:

* Strategy returns
* Factor returns
* Rolling window configuration
* Optional factor column selections

It exports:

* `rolling_factor_exposures.csv`
* `rolling_factor_exposure_summary.csv`

## Consequences

Rolling factor exposure analysis becomes accessible from the command line.

This supports portfolio research workflows that need to inspect how strategy betas, alpha, residual volatility, and dominant factor exposures evolve over time.

Future work can integrate rolling exposure tables directly into the advanced research export package and memo.
