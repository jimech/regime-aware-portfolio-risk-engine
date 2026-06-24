# ADR 0085: Advanced research package loader

## Status

Accepted

## Context

The advanced research package now includes a `manifest.json` file and a CLI command for package inspection.

Future notebooks and dashboards need a reusable Python interface for loading a generated package into memory.

## Decision

Add an advanced research package loader.

The loader reads the package manifest, loads the Markdown memo as text, and loads all exported CSV tables into pandas DataFrames keyed by logical table name.

## Consequences

Generated research packages can now be consumed programmatically by notebooks, dashboards, and downstream analysis tools.

This creates a clean bridge toward a Streamlit dashboard without coupling dashboard code to the package export internals.