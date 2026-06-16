# ADR 0035: CLI config inspection

## Status

Accepted

## Context

The project relies on YAML configuration for assets, dates, paths, and workflow settings.

Users need a simple way to verify that a config file is readable and contains the expected high-level settings before running longer workflows.

## Decision

Add a CLI config inspection command.

The command is:

```bash
regime-risk-engine inspect-config