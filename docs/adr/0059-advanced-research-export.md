# ADR 0059: Advanced research export package

## Status

Accepted

## Context

The advanced research memo combines multiple portfolio diagnostics into one investment committee style document.

However, reviewers may also need access to the underlying tables used to support the memo.

## Decision

Add an advanced research export package.

The package exports the advanced research memo and all available supporting tables, including:

- Regime intelligence profile
- Regime transition tables
- Stress-test summary
- Attribution tables
- Factor exposure tables
- Scenario simulation outputs

## Consequences

The project can now generate a full advanced research deliverable folder.

This improves reproducibility, reviewability, and professional presentation quality.

Future work can add a CLI command that runs the full advanced workflow and exports the package in one step.