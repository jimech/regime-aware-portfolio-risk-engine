# ADR 0087: Advanced research package summary helper

## Status

Accepted

## Context

The project now includes a loader for advanced research packages.

Future dashboards and notebooks need a compact way to summarize package contents without manually inspecting every table.

## Decision

Add a dashboard-ready package summary helper.

The helper extracts the memo title, counts tables, lists table names, and identifies whether key research diagnostics are present.

## Consequences

Advanced research packages become easier to display in dashboards and notebooks.

This creates a lightweight bridge from package loading to the future Streamlit dashboard.