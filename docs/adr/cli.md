# Command-line interface

The project includes a command-line interface for inspecting configs, checking the package, and generating report artifacts.

You can run commands in two ways.

Use the module form during local development:

```bash
python -m regime_risk_engine healthcheck

## Create advanced demo inputs

The `create-advanced-demo-inputs` command creates deterministic CSV inputs for the advanced research export workflow.

It writes:

- `prices.csv`
- `static_weights.csv`
- `regime_policy.csv`
- `stress_periods.csv`
- `factor_returns.csv`

Example:

```bash
python -m regime_risk_engine create-advanced-demo-inputs \
  --output-dir outputs/demo_inputs