# ADR 0082: Advanced research package manifest documentation

## Status

Accepted

## Context

The advanced research package now writes a machine-readable `manifest.json` file.

Users and reviewers need documentation explaining what the manifest contains and why it is useful.

## Decision

Add an example document describing the advanced research package manifest.

The document explains:

- The purpose of `manifest.json`
- The expected top-level fields
- Example manifest structure
- How the manifest supports reviewability and downstream tooling

## Consequences

The advanced research package is easier to inspect and consume.

The project documentation now explains the package structure in a way that supports future notebooks, dashboards, and automated validation.