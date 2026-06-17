# ADR 0048: Market research export package

## Status

Accepted

## Context

The market research workflow and memo builder can produce investment research outputs in memory.

For the project to be useful as a portfolio research tool, those outputs need to be exported as files that can be reviewed, shared, or included in a project portfolio.

## Decision

Add a market research export package.

The export package writes:

- Investment memo
- Strategy metric table
- Metric delta table
- Regime metric table
- Regime summary table
- Dynamic target weights
- Dynamic applied weights
- Regime labels

## Consequences

The project can now produce a complete research deliverable folder.

This supports future CLI workflows, reproducible research runs, and portfolio-manager-style review.