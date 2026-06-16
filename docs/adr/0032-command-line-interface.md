# ADR 0032: Command-line interface

## Status

Accepted

## Context

The project has grown from individual analytics modules into a larger risk engine with data, features, regime detection, allocation, backtesting, validation, and reporting layers.

Users should be able to interact with the project from the terminal without writing Python code for every basic operation.

## Decision

Implement a lightweight command-line interface.

The first CLI layer supports:

- Package version checks
- Basic project healthchecks
- Optional output directory validation/creation

The CLI entry point is:

```bash
regime-risk-engine