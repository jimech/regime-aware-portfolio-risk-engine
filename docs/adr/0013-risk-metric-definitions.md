# ADR 0013: Risk metric definitions

## Status

Accepted

## Context

The project needs consistent risk and performance metrics for comparing static and dynamic portfolio strategies.

The same definitions should be used across regime-level analytics, backtesting, validation, reports, and dashboards.

## Decision

Implement a reusable metrics module with the following metrics:

- Cumulative return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Historical Value at Risk
- Historical Conditional Value at Risk

The first implementation assumes simple periodic returns.

VaR and CVaR are reported as positive loss values.

The annualization factor is configurable and defaults to 252 trading days.

The risk-free rate used in Sharpe ratio is interpreted as an annual rate.

## Consequences

Metric definitions are centralized and reusable.

This reduces the risk of inconsistent calculations across the project.

Historical VaR and CVaR are simple and transparent, but they depend heavily on the observed return sample and do not assume a parametric return distribution.

## Regime-specific risk summaries

Risk metrics are also calculated within each detected regime.

The project supports both portfolio-level and asset-level regime risk summaries.

Portfolio returns can be calculated using equal weights or explicit user-provided weights.

Returns and regime labels are aligned by date before metrics are calculated.

This allows the project to answer questions such as:

- Which regimes have the highest volatility?
- Which regimes have the worst drawdowns?
- Which assets perform best or worst in each regime?
- Does tail risk increase during specific regimes?