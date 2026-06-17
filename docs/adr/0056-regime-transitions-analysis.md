# ADR 0056: Regime transition analysis

## Status

Accepted

## Context

Detected regimes are more useful when their stability and transition behavior are understood.

A portfolio manager needs to know whether regimes are persistent, unstable, or likely to rotate into specific future states.

Numeric regime labels alone do not describe transition risk.

## Decision

Add regime transition analysis.

The module estimates:

- Transition count matrix
- Transition probability matrix
- Regime persistence probabilities
- Expected regime duration
- Consecutive regime duration blocks
- Most likely next regime by current regime
- Narrative summary of regime stability

## Consequences

The project can now evaluate whether detected market regimes behave like coherent market states.

This improves regime model diagnostics and supports more realistic allocation research.

Future work can add Markov-chain simulation, regime probability forecasting, and transition-aware allocation.