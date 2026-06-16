# ADR 0033: CLI report export command

## Status

Accepted

## Context

The project now supports report tables, report plots, report exports, and report assembly from Python.

Users also need a terminal workflow for exporting report artifacts from saved CSV tables and existing figure files.

## Decision

Add a CLI report export command.

The command is:

```bash
regime-risk-engine export-report