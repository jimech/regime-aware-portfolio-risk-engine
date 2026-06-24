# ADR 0084: Advanced package manifest inspection CLI

## Status

Accepted

## Context

The advanced research package now includes a `manifest.json` file that records the generated memo and exported tables.

Users need a simple way to inspect and validate a package from the command line.

## Decision

Add an `inspect-advanced-package` CLI command.

The command reads `manifest.json`, validates referenced files, and prints the memo and table paths. It also supports JSON output for automated tooling.

## Consequences

Advanced research packages become easier to review, debug, and consume programmatically.

This also provides a foundation for future dashboard or notebook tooling that loads packages from their manifest.