# ADR 0011: Probabilistic regime model

## Status

Accepted

## Context

The baseline K-Means regime model assigns each date to one hard cluster.

Market regimes may not always be cleanly separated. Transitions between regimes can be gradual, and some periods may contain characteristics of multiple regimes.

The project needs a model that can provide both hard labels and soft regime probabilities.

## Decision

Add a Gaussian Mixture Model as the first probabilistic regime detection model.

The model accepts the date-indexed feature matrix and produces:

- Hard regime labels
- Regime probability estimates

The number of regimes is configurable.

The first implementation uses a fixed random seed for reproducibility.

## Consequences

Gaussian Mixture Models provide more information than hard clustering because each date receives a probability distribution over regimes.

This can help identify uncertain transition periods.

However, GMMs still have limitations:

- They do not explicitly model temporal persistence.
- They can be sensitive to scaling and initialization.
- Regime labels remain numeric and require a later interpretation layer.
- Probability estimates should not be interpreted as true economic probabilities without validation.

Future work may add Hidden Markov Models or other time-aware regime detection methods.