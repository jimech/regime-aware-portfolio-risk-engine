# ADR 0075: Rolling factor exposure advanced demo integration

## Status

Accepted

## Context

Rolling factor exposure analysis is available as a standalone research module and CLI export.

The advanced research demo already passes factor returns through the advanced workflow, which enables rolling factor exposure estimation when factor data is available.

## Decision

Confirm and test that the advanced demo package exports rolling factor exposure outputs.

The advanced demo package includes:

- `rolling_factor_exposures.csv`
- `rolling_factor_exposure_summary.csv`

The advanced research memo also includes a `Rolling Factor Exposure Analysis` section.

## Consequences

The advanced demo now demonstrates time-varying factor exposure diagnostics as part of the main investment research package.

This strengthens the project’s thesis by showing not only whether the dynamic strategy performed well, but whether its factor risk exposures evolved through time.