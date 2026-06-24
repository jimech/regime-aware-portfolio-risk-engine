# Advanced Research Package Loader

The advanced research package loader reads a generated package from its `manifest.json`.

It loads the Markdown memo as text, all exported CSV tables as pandas DataFrames, and the package metadata.

## Generate a package

Run this command:

python -m regime_risk_engine run-advanced-demo --output-dir outputs/advanced_demo --analyst "Jimena Chinchilla"

The package will be written to `outputs/advanced_demo/package`.

## Load the package in Python

Import `load_advanced_research_package` from `regime_risk_engine.research.package_manifest`.

Example usage:

package = load_advanced_research_package("outputs/advanced_demo/package")

Then inspect:

- package.memo
- package.table_names()
- package.tables

## Access tables

Common table keys include:

- factor_significance
- rolling_factor_exposures
- scenario_terminal_summary
- stress_test_summary
- regime_intelligence_profile

Each table is loaded as a pandas DataFrame.

## Why this matters

The loader lets notebooks, dashboards, and downstream analysis tools consume an advanced research package without hard-coding every CSV filename.

The package manifest remains the source of truth for what files were generated.

## Research use

The loader is useful for notebook review workflows, Streamlit dashboards, automated package checks, custom research reports, and follow-up exploratory analysis.
