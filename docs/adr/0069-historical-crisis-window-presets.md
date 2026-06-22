# ADR 0068: Historical crisis window presets

## Status

Accepted

## Context

The project includes stress-period analysis, but stress windows should not only be synthetic or manually created for each run.

For investment research, named historical market stress periods make results easier to interpret and review. They also make the project more credible because strategy behavior can be evaluated against recognizable market environments.

## Decision

Add reusable historical crisis window presets.

The initial preset library includes:

* Global Financial Crisis
* Eurozone debt crisis
* Volmageddon
* Q4 2018 selloff
* COVID crash
* 2022 inflation and interest-rate shock
* 2023 regional bank stress

The presets are stored as structured records with:

* Name
* Start date
* End date
* Description
* Category

The module also supports exporting presets as CSV and converting them into stress-period input format.

## Consequences

Stress testing becomes easier to reuse across workflows.

Research outputs can refer to recognizable historical events instead of unnamed stress windows.

Future work can connect these presets directly into the advanced demo workflow and allow users to select crisis windows from the CLI.
