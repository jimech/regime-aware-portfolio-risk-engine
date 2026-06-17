# ADR 0063: Advanced demo input CLI

## Status

Accepted

## Context

The advanced research workflow can be run from CSV inputs, and deterministic demo inputs can be generated from Python.

However, users should be able to create those demo inputs directly from the command line.

## Decision

Add a CLI command for advanced demo input generation.

The command creates CSV files for:

- Price data
- Static benchmark weights
- Regime-aware allocation policy
- Stress periods
- Factor returns

## Consequences

The full advanced research workflow is easier to demo from the terminal.

This improves usability and makes the project more reviewable as an end-to-end portfolio analytics engine.

Future work can add a single command that creates demo inputs and immediately runs the advanced export.