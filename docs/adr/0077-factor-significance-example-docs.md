# ADR 0077: Factor significance example documentation

## Status

Accepted

## Context

The advanced research package now exports factor significance diagnostics and includes a factor significance section in the advanced memo.

Users need documentation explaining how to interpret beta, standard error, t-statistic, p-value, and significance flags.

## Decision

Add a factor significance example document under `docs/examples`.

The document explains:

- The purpose of factor significance diagnostics
- The exported CSV columns
- How to interpret significant and non-significant betas
- How the feature supports regime-aware research

## Consequences

The factor significance output becomes easier to understand for reviewers and users.

The project’s research package now documents not only factor exposures, but also the statistical evidence behind those exposures.