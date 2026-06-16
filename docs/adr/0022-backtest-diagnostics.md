# ADR 0022: Backtest diagnostics

## Status

Accepted

## Context

Final strategy metrics are useful, but they do not show how performance and risk evolve over time.

The project needs diagnostic time series to inspect strategy behavior during different market environments.

## Decision

Implement reusable backtest diagnostics.

The first diagnostic layer includes:

- Cumulative return series
- Drawdown series
- Rolling annualized volatility
- Rolling annualized Sharpe ratio

Diagnostics can be calculated for one or more aligned strategy return series.

## Consequences

The project can inspect whether the dynamic regime-aware strategy improves performance consistently or only during specific periods.

Drawdown diagnostics help identify the timing and severity of losses.

Rolling volatility and rolling Sharpe diagnostics help evaluate strategy stability through time.

The rolling metrics depend on the selected window length, so results should be interpreted alongside the chosen window.