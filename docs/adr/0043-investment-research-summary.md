# ADR 0043: Investment research summary layer

## Status

Accepted

## Context

The project can now calculate strategy metrics, regime-conditioned metrics, model diagnostics, reports, and CLI outputs.

However, analytical tables alone do not explain whether the regime-aware strategy is actually useful from an investment perspective.

The project needs an interpretation layer that converts metrics into professional research conclusions.

## Decision

Add an investment research summary layer.

The first version interprets:

- Candidate strategy versus benchmark metric deltas
- Whether each metric moved in a favorable or unfavorable direction
- Regime-level candidate versus benchmark performance
- Overall strategy verdict
- Executive research summary text

The layer treats higher values as favorable for returns and risk-adjusted metrics.

The layer treats lower values as favorable for volatility, drawdown, VaR, CVaR, turnover, and transaction costs.

## Consequences

The project begins moving from engineering infrastructure into actual investment research.

Analytical outputs can now be translated into portfolio-management conclusions.
