# ADR 0027: Regime model selection

## Status

Accepted

## Context

The project may evaluate multiple regime detection models or configurations.

Walk-forward validation produces split-level diagnostics, but the project needs a clear summary layer to compare candidates.

Model selection should remain transparent because regime detection is unsupervised and no single metric proves economic usefulness.

## Decision

Implement a regime model selection summary layer.

The first comparison layer summarizes each walk-forward validation result using:

- Split count
- Mean train and test observation counts
- Mean train and test regime counts
- Mean train and test transition rates
- Mean train and test dominant regime shares
- Mean train and test silhouette scores
- Mean train and test Calinski-Harabasz scores

The layer also supports ranking models by a selected metric.

The default ranking metric is test silhouette score.

## Consequences

Candidate regime models can be compared consistently.

The ranking is transparent and configurable.

Model selection should not rely on this ranking alone. Final model choice should also consider regime interpretability, regime-specific risk behavior, and downstream backtest performance.