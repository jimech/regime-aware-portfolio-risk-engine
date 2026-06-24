# ADR 0083: Advanced package manifest integrity tests

## Status

Accepted

## Context

The advanced research package now includes a machine-readable `manifest.json`.

The manifest should be trustworthy because downstream tools, notebooks, dashboards, and reviewers may rely on it to locate package artifacts.

## Decision

Add integrity tests for the advanced research package manifest.

The tests verify that:

- The manifest memo entry points to the generated memo.
- Every manifest table points to an existing package file.
- The manifest includes every exported table.
- Manifest paths are relative rather than absolute.

## Consequences

The package manifest is protected against regressions.

This makes the advanced research package easier to consume programmatically and safer to share across machines.