# ADR 0074: Rolling factor exposure example documentation

## Status

Accepted

## Context

The project now includes rolling factor exposure analysis and a CLI export command.

However, users need documentation showing what input files are expected, how to run the command, what files are exported, and how to interpret the results.

## Decision

Add an example document for rolling factor exposure analysis.

The document explains:

- Strategy return input format
- Factor return input format
- CLI usage
- Optional CLI arguments
- Exported CSV files
- Interpretation of rolling betas
- How rolling exposure supports regime-aware research

## Consequences

Rolling factor exposure analysis becomes easier to understand and use.

The project’s documentation now shows how factor betas can be used to evaluate whether strategy risk exposures change over time.

Future work can integrate rolling factor exposure tables directly into the advanced research export and memo.
