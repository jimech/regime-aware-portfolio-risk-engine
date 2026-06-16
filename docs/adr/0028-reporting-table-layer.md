# ADR 0028: Reporting table layer

## Status

Accepted

## Context

The project produces many analytical outputs, including backtest metrics, regime-conditioned metrics, model validation diagnostics, and model selection rankings.

These outputs need to be converted into consistent report-ready tables for notebooks, dashboards, and final written reporting.

## Decision

Implement a reporting table layer.

The first reporting layer creates tables for:

- Strategy metric summaries
- Candidate-minus-benchmark metric deltas
- Regime-level strategy metrics
- Regime model rankings

Metric names are converted into readable labels while preserving machine-readable metric identifiers.

## Consequences

The project has a clean boundary between analytics and presentation.

Analytics modules can continue returning computational outputs, while reporting modules reshape those outputs for users.

Future reporting work can build charts, dashboards, Markdown exports, HTML exports, or PDF reports on top of this table layer.