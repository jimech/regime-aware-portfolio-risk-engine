# ADR 0023: Regime-conditioned backtest evaluation

## Status

Accepted

## Context

The project compares a static benchmark strategy against a dynamic regime-aware strategy.

Overall backtest metrics show aggregate performance, but they do not explain which market regimes drive outperformance or underperformance.

The project needs regime-conditioned backtest evaluation to answer whether dynamic allocation helps during specific detected regimes.

## Decision

Implement regime-conditioned strategy evaluation.

The evaluation layer aligns:

- Strategy return series
- Date-indexed regime labels

For each regime and strategy, calculate the standard risk metric summary.

Then calculate candidate-minus-benchmark metric deltas within each regime.

The default comparison treats:

- Static strategy as benchmark
- Dynamic strategy as candidate

## Consequences

The project can explain where the dynamic strategy adds or loses value.

This supports regime-aware interpretation beyond aggregate backtest performance.

Results depend on the number of observations per regime. Regimes with few observations should be interpreted carefully.