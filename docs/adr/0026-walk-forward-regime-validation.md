# ADR 0026: Walk-forward regime validation

## Status

Accepted

## Context

The project uses unsupervised regime detection models on financial time-series features.

Models should be validated chronologically so that future information is not used when fitting models intended to predict future regimes.

The project needs a walk-forward validation layer that re-fits models on past data and evaluates regime predictions on future data.

## Decision

Implement walk-forward regime model validation.

For each chronological validation split:

- Create a fresh model instance
- Fit on train features
- Predict train regime labels
- Predict test regime labels
- Store predicted labels
- Calculate regime-count and transition diagnostics
- Calculate internal clustering diagnostics when valid

The first internal diagnostics are:

- Silhouette score
- Calinski-Harabasz score

These diagnostics are only calculated when there are at least two regimes and fewer regimes than observations.

## Consequences

Regime models can be evaluated in a way that better matches real deployment.

The validation output shows how regime behavior changes across walk-forward splits.

Internal clustering metrics provide one diagnostic layer but do not prove economic usefulness.

Economic usefulness must still be evaluated through regime risk summaries, allocation behavior, and backtest performance.