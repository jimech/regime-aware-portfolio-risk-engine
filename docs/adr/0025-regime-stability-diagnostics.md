# ADR 0025: Regime stability diagnostics

## Status

Accepted

## Context

Regime detection is unsupervised, so model validation cannot rely only on traditional classification accuracy.

Detected regime labels may be arbitrary, unstable across model runs, or too noisy to support allocation decisions.

The project needs diagnostics that evaluate regime stability and interpretability.

## Decision

Implement regime stability diagnostics.

The first validation layer includes:

- Regime distribution
- Regime transition matrix
- Regime transition summary
- Label agreement between model outputs
- Multi-model stability reports

Label agreement uses:

- Direct match rate
- Adjusted Rand Index
- Normalized mutual information

Adjusted Rand Index and normalized mutual information are label-invariant, which is important because unsupervised regime IDs may be permuted between model runs.

## Consequences

The project can evaluate whether regimes are stable enough to support downstream allocation.

Transition diagnostics show whether regimes switch too frequently.

Distribution diagnostics show whether one regime dominates the sample.

Agreement diagnostics help compare model runs, configurations, or validation splits.

These diagnostics do not prove that regimes are economically meaningful. They are one validation layer that should be combined with risk summaries, backtests, and interpretability checks.