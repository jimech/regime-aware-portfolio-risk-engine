# ADR 0045: Investment research pipeline

## Status

Accepted

## Context

The project can calculate backtest comparisons, regime-conditioned evaluations, reporting tables, and investment research summaries.

These pieces need to be connected into one research pipeline that produces report-ready analytical tables and investment interpretation text.

## Decision

Add an investment research pipeline runner.

The first pipeline consumes:

- Static versus dynamic strategy comparison results
- Regime-conditioned backtest evaluation results

It produces:

- Strategy metric report table
- Metric delta report table
- Regime metric report table
- Strategy research summary
- Regime research summary
- Executive research summary

## Consequences

The project now has a clean path from quantitative outputs to investment conclusions.

This establishes the layer that will later support real historical market narratives, final research reports, and investment committee style summaries.