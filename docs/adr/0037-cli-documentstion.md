# ADR 0037: CLI documentation

## Status

Accepted

## Context

The project now includes several command-line workflows.

The CLI supports healthchecks, config inspection, report exports, demo input generation, and full demo report generation.

Users need a single documentation page that explains how to run these commands during local development and after package installation.

## Decision

Add dedicated CLI documentation.

The documentation covers:

- Module-based invocation
- Console-script invocation
- Version command
- Healthcheck command
- Config inspection command
- Demo report input generation
- Report export
- Full demo report workflow
- Cleanup guidance
- Troubleshooting

A lightweight test verifies that the CLI documentation exists and includes the supported commands.

## Consequences

Users have a clear reference for CLI usage.

The documentation supports both development and installed-package workflows.

The test suite now guards against accidentally removing CLI documentation.