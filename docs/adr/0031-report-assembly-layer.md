# ADR 0031: Report assembly layer

## Status

Accepted

## Context

The project now has separate modules for analytics, diagnostics, report tables, report plots, and report exports.

A final orchestration layer is needed so users can create a complete report from standard project outputs without manually wiring every table and figure together.

## Decision

Implement a report assembly layer.

The assembly layer consumes:

- Static vs dynamic strategy comparison results
- Backtest diagnostics
- Regime-conditioned evaluation results
- Regime model selection results

It produces:

- Report-ready tables
- Report-ready figures
- Optional exported report artifacts

## Consequences

The project now has a clean end-to-end reporting workflow.

Analytics modules remain separate from presentation and export logic.

The report assembly layer provides a single high-level entry point for notebooks, dashboards, and final project reporting.