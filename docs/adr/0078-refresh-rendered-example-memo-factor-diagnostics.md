# ADR 0078: Refresh rendered example memo with factor diagnostics

## Status

Accepted

## Context

The advanced research memo now includes rolling factor exposure and factor significance diagnostics.

The checked-in rendered example memo should reflect the current advanced research output so reviewers can see the full research package without running the demo locally.

## Decision

Refresh `docs/examples/advanced_research_memo_example.md` from a newly generated advanced demo package.

The refreshed memo includes:

- Rolling factor exposure analysis
- Factor significance analysis
- Regression alpha, R-squared, observation count, and p-value diagnostics

## Consequences

The rendered example memo now matches the current advanced research workflow more closely.

This improves project reviewability because the example output demonstrates both time-varying factor exposure diagnostics and statistical evidence checks.