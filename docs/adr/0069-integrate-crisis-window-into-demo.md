# ADR 0069: Integrate crisis windows into advanced demo inputs

## Status

Accepted

## Context

The project includes reusable historical crisis window presets and an advanced demo workflow.

Before this change, the advanced demo could generate stress-period inputs, but those periods were not directly connected to the reusable crisis-window preset library.

For investment research, recognizable historical stress windows make demo outputs easier to interpret and more credible.

## Decision

Update the advanced demo input generator so `stress_periods.csv` is built from the reusable historical crisis-window presets when those presets overlap the generated demo price index.

If no crisis preset overlaps the generated demo data, the workflow falls back to a deterministic synthetic demo stress window.

## Consequences

The advanced demo becomes more realistic and easier to explain.

Stress-period analysis can now reference recognizable crisis labels such as COVID crash, inflation and rate shock, and regional bank stress when the demo date range supports them.

Future work can expose crisis-window selection through the CLI.

