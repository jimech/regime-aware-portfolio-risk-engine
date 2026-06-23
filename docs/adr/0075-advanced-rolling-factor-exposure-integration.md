# ADR 0075: Advanced rolling factor exposure integration

## Status

Accepted

## Context

Rolling factor exposure analysis was added as a standalone research capability and CLI export.

The advanced research package already exports regime intelligence, transition analysis, stress testing, attribution, static factor exposure, and scenario simulation outputs. However, rolling factor exposure was not yet included in the main advanced research package.

## Decision

Integrate rolling factor exposure into the advanced research workflow when factor returns are supplied.

The advanced workflow now estimates rolling factor exposure for the dynamic strategy. The advanced memo includes a rolling factor exposure section, and the advanced export package writes:

- `rolling_factor_exposures.csv`
- `rolling_factor_exposure_summary.csv`

## Consequences

The advanced research package now explains not only average factor exposure, but also how factor betas changed over time.

This strengthens the project thesis by connecting regime-aware allocation decisions to observable changes in dynamic strategy risk exposure.
