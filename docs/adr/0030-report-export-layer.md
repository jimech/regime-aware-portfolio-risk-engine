# ADR 0030: Report export layer

## Status

Accepted

## Context

The project produces report-ready tables and figures.

These artifacts need to be saved in a consistent, reproducible folder structure for notebooks, dashboards, reviews, and final project reporting.

## Decision

Implement a report export layer.

The first export layer supports:

- CSV exports for report tables
- PNG exports for report figures
- Markdown report index
- JSON artifact manifest

The export folder uses subdirectories:

```text
tables/
figures/