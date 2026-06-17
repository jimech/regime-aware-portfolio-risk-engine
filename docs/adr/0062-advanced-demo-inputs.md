# ADR 0062: Advanced research demo inputs

## Status

Accepted

## Context

The advanced research CLI can export a full investment research package from CSV inputs.

However, manually creating price data, static weights, regime policy, stress periods, and factor return files makes the workflow harder to demo.

A professional project should include deterministic demo inputs that make the advanced workflow easy to run.

## Decision

Add an advanced research demo input generator.

The generator creates:

- Price data
- Static benchmark weights
- Regime-aware allocation policy
- Stress-period definitions
- Factor return series

The demo uses three intuitive market environments:

- Growth expansion
- Defensive drawdown
- Inflation rotation

## Consequences

The advanced research workflow can now be demonstrated quickly using generated inputs.

This improves usability, reviewability, and portfolio-project presentation quality.

Future work can add a CLI command that creates these demo inputs directly from the terminal.