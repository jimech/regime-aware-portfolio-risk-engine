# ADR 0036: CLI demo report workflow

## Status

Accepted

## Context

The command-line interface supports creating demo report inputs and exporting report artifacts.

Users need a single command that runs the full demo workflow end-to-end.

## Decision

Add a CLI demo report workflow command.

The command is:

```bash
regime-risk-engine run-demo-report