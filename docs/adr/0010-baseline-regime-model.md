# ADR 0010: Baseline regime model

## Status

Accepted

## Context

The project needs an initial unsupervised model for detecting market regimes from engineered financial features.

The first model should be simple, reproducible, easy to test, and easy to explain before more advanced models are added.

## Decision

Use K-Means clustering as the first baseline regime detection model.

The model accepts the date-indexed regime feature matrix and assigns each date to a regime label.

The number of regimes is configurable.

The first version uses a fixed random seed for reproducibility.

## Consequences

K-Means provides a useful baseline for regime detection.

It is simple and fast, making it suitable for early development and testing.

However, K-Means has limitations:

- It assumes roughly spherical clusters.
- It does not model regime transition probabilities.
- It does not account for temporal persistence.
- Regime labels are numeric and require a later interpretation layer.

Future tickets may add Gaussian Mixture Models, Hidden Markov Models, and regime labeling logic.