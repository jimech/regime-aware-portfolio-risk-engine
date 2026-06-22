# ADR 0070: Demo stress-period mode CLI option

## Status

Accepted

## Context

The advanced demo workflow now supports historical crisis-window presets.

However, users may want to distinguish between realistic historical stress periods and deterministic synthetic demo-only stress periods.

The CLI should make this choice explicit instead of hiding it inside the input generator.

## Decision

Add a `--stress-period-mode` option to advanced demo commands.

Supported values are:

* `crisis`: use historical crisis-window presets when they overlap the generated demo data.
* `synthetic`: use a deterministic synthetic demo-only stress window.

The default mode is `crisis`.

## Consequences

The advanced demo workflow becomes more transparent and configurable.

Users can generate research packages using recognizable historical stress labels or simpler deterministic demo-only stress windows.

Future work can expose named crisis-window selection and support user-provided stress-period presets directly from the demo workflow.
