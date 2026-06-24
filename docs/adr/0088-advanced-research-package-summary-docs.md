# ADR 0088: Advanced research package summary documentation

## Status

Accepted

## Context

The project now includes a dashboard-ready summary helper for loaded advanced research packages.

Users and reviewers need documentation explaining what the helper returns and how it supports notebooks and dashboards.

## Decision

Add documentation for the advanced research package summary helper.

The documentation explains:

- How to generate a package
- How to load and summarize it
- Which summary fields are returned
- How dashboards can use the feature flags

## Consequences

The package summary helper becomes easier to discover and use.

This prepares the project for a future dashboard interface while keeping the package-loading layer independent from Streamlit.