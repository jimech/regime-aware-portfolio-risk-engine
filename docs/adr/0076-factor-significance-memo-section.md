# ADR 0076: Factor significance memo section

## Status

Accepted

## Context

The advanced research workflow estimates factor significance diagnostics when factor return inputs are provided.

The advanced export package already writes `factor_significance.csv`, but the advanced memo did not summarize these diagnostics directly.

## Decision

Add a `Factor Significance Analysis` section to the advanced research memo.

The section reports:

- Factor beta
- Standard error
- T-statistic
- P-value
- Significance flag
- Regression alpha
- Regression R-squared
- Observation count

## Consequences

The advanced memo now distinguishes economically large factor exposures from statistically noisy relationships.

This improves the research package by adding evidence about whether estimated factor betas are meaningfully different from zero.