# ADR 0060: Advanced research workflow

## Status

Accepted

## Context

The project now includes multiple advanced research modules, including regime intelligence, transition analysis, stress testing, attribution, factor exposure, scenario simulation, and advanced memo generation.

These modules are useful individually, but users need a simple workflow that builds the full advanced research package from a market workflow result.

## Decision

Add an advanced research workflow builder.

The workflow consumes a market research workflow result and optionally accepts stress periods and factor returns.

It builds:

- Base market research memo
- Regime intelligence summary
- Regime transition summary
- Stress-period summary, when stress periods are supplied
- Strategy attribution summary
- Factor exposure summary, when factor returns are supplied
- Forward regime scenario simulation
- Advanced research memo

## Consequences

The project can now generate an integrated advanced research memo from the core market workflow output.

This reduces manual glue code and makes the research system easier to use as an end-to-end portfolio analytics engine.

Future work can add CLI support for this full advanced workflow.