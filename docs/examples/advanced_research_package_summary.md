# Advanced Research Package Summary

The advanced research package summary helper creates a compact, dashboard-ready overview of a loaded research package.

It is useful for notebooks, dashboards, automated checks, and portfolio review workflows.

## Generate a package

Run:

python -m regime_risk_engine run-advanced-demo --output-dir outputs/advanced_demo --analyst "Jimena Chinchilla"

The generated package will be written to:

outputs/advanced_demo/package

## Load and summarize the package

Import the package loader and summary helper:

- load_advanced_research_package
- summarize_advanced_research_package

Example usage:

package = load_advanced_research_package("outputs/advanced_demo/package")
summary = summarize_advanced_research_package(package)

The summary includes:

- memo_title
- table_count
- table_names
- has_factor_significance
- has_rolling_factor_exposure
- has_scenario_simulation
- has_stress_test

## Example output

The summary can be converted to a dictionary:

summary.to_dict()

Example fields:

- memo_title: Advanced Regime-Aware Portfolio Research Memo
- table_count: number of exported CSV tables
- has_factor_significance: true when factor significance diagnostics are present
- has_rolling_factor_exposure: true when rolling factor exposure outputs are present
- has_scenario_simulation: true when scenario simulation tables are present
- has_stress_test: true when stress-period analysis is present

## Dashboard use

A dashboard can use this summary to decide which sections to display.

For example:

- Show factor significance charts only when has_factor_significance is true.
- Show rolling beta tables only when has_rolling_factor_exposure is true.
- Show scenario simulation outputs only when has_scenario_simulation is true.
- Show stress-period diagnostics only when has_stress_test is true.

## Research use

The package summary provides a quick overview of which research diagnostics are available in a generated package.

It does not replace the full memo or exported tables. It is a lightweight index for user interfaces and review workflows.