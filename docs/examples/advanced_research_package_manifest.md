# Advanced Research Package Manifest

The advanced research package includes a machine-readable manifest file named `manifest.json`.

The manifest lists the generated memo and supporting research tables in the package.

## Why this matters

The advanced research package contains many outputs, including regime diagnostics, stress tests, attribution, factor exposure analysis, rolling factor exposure, factor significance diagnostics, and scenario simulation tables.

The manifest makes the package easier to inspect programmatically.

## Example structure

Example manifest fields:

- "memo": "advanced_research_memo.md"
- "tables": mapping of logical table names to CSV filenames

Example table entries:

- "asset_attribution": "asset_attribution.csv"
- "dominant_factor_by_strategy": "dominant_factor_by_strategy.csv"
- "factor_exposure": "factor_exposure.csv"
- "factor_significance": "factor_significance.csv"
- "rolling_factor_exposure_summary": "rolling_factor_exposure_summary.csv"
- "rolling_factor_exposures": "rolling_factor_exposures.csv"
- "scenario_terminal_summary": "scenario_terminal_summary.csv"
- "stress_test_summary": "stress_test_summary.csv"

## Fields

| Field | Meaning |
| --- | --- |
| `memo` | The generated Markdown research memo filename. |
| `tables` | Mapping of logical table names to exported CSV filenames. |

## Research use

The manifest helps downstream tools, notebooks, dashboards, and reviewers locate package artifacts without hard-coding every filename.

It also improves reproducibility by making the package contents explicit.
