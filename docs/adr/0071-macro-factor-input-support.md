# ADR 0071: Macro factor input support

## Status

Accepted

## Context

The project already supports factor exposure analysis, but demo factors are currently simple strategy-style factors such as equity, defensive, and real asset returns.

To make regime analysis more economically meaningful, the research engine should support macroeconomic inputs that can describe market environments.

Examples include:

* Inflation
* Policy rates
* Yield curve slope
* Credit spreads
* Dollar strength
* Commodity index levels
* Growth proxies

These inputs often arrive as levels rather than returns, so they need configurable transformations before being used in research workflows.

## Decision

Add a macro factor input module.

The module supports:

* Level transformations
* First differences
* Percent changes
* Log differences
* Lagged factors
* Optional standardization
* CSV reading and writing
* Automatic factor spec inference from numeric columns

The module produces a model-ready factor matrix with a `date` column and transformed factor columns.

## Consequences

The engine can support more economically interpretable regime research.

Future workflows can connect macro factor matrices into factor exposure analysis, regime intelligence, and advanced research exports.

This keeps macro processing separate from live data fetching and avoids turning the project into a real-time macro data system.
