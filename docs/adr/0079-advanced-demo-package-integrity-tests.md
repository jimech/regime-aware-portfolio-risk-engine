# ADR 0079: Advanced demo package integrity tests

## Status

Accepted

## Context

The advanced demo workflow generates a complete research package containing a memo and many supporting CSV tables.

As the package grows, regressions could accidentally remove, empty, or rename important outputs without being caught by narrow feature tests.

## Decision

Add package integrity tests for the advanced demo workflow.

The tests verify that:

- Exported table paths exist.
- Exported table files are non-empty.
- Exported CSV files contain rows.
- The package includes expected core research artifacts.
- The advanced memo references the core research sections.

## Consequences

The advanced demo package becomes more reliable and reviewable.

This protects the project’s main portfolio artifact from silent regressions as new research features are added.