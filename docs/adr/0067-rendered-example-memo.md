# ADR 0067: Rendered example memo documentation

## Status

Accepted

## Context

The project can generate advanced investment research memos from the command line.

However, reviewers may not always run the CLI before evaluating the repository. A rendered example memo gives readers an immediate understanding of the project’s final research output.

## Decision

Add a documented example advanced research memo under `docs/examples/`.

The memo is illustrative and explains the structure of the generated output, including:

* Executive summary
* Research question
* Regime intelligence
* Regime transition analysis
* Static versus dynamic comparison
* Stress-period analysis
* Strategy attribution
* Factor exposure analysis
* Scenario simulation
* Limitations

## Consequences

The project becomes easier to evaluate from the GitHub page alone.

Readers can quickly understand the type of investment research artifact produced by the engine without running the full workflow.

Future work can add a fully generated memo artifact from a real historical dataset.
