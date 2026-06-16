# ADR 0024: Time-series validation splits

## Status

Accepted

## Context

The project uses regime detection models on financial time-series data.

Random train/test splits are inappropriate because they can leak future information into the training set.

The project needs chronological validation splits for model evaluation.

## Decision

Implement reusable time-series validation split utilities.

The first implementation supports:

- Expanding-window splits
- Rolling-window splits
- Configurable train size
- Configurable test size
- Configurable step size
- Optional maximum number of splits

Train dates must always occur before test dates.

Train and test dates cannot overlap.

## Consequences

Regime models can be evaluated more realistically.

The validation layer reduces the risk of look-ahead leakage.

Expanding windows use all available past history, while rolling windows keep a fixed training length.

Later validation tickets can reuse these splits for regime prediction stability, backtest robustness, and walk-forward strategy evaluation.