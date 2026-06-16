# CLI Reference

The Regime Risk Engine command-line interface is available after installing the
project in editable mode:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run commands through the installed console script:

```bash
regime-risk-engine --help
```

You can also invoke the CLI as a Python module:

```bash
python -m regime_risk_engine --help
```

## Commands

### version

Print the package version.

```bash
python -m regime_risk_engine version
```

### healthcheck

Run a basic import and project healthcheck.

```bash
python -m regime_risk_engine healthcheck
python -m regime_risk_engine healthcheck --json
python -m regime_risk_engine healthcheck --output-dir reports
```

### inspect-config

Inspect a YAML project configuration file and summarize detected settings.

```bash
python -m regime_risk_engine inspect-config --config configs/base.yaml
python -m regime_risk_engine inspect-config --config configs/base.yaml --json
```

### create-demo-report-inputs

Create demo CSV tables and PNG figures that can be passed to `export-report`.

```bash
python -m regime_risk_engine create-demo-report-inputs --output-dir reports/demo-inputs
python -m regime_risk_engine create-demo-report-inputs --output-dir reports/demo-inputs --json
```

### export-report

Export report-ready CSV tables and optional PNG figures to a report directory.
Table and figure inputs use `name=path` pairs.

```bash
python -m regime_risk_engine export-report \
  --output-dir reports/demo-report \
  --table strategy_metrics=reports/demo-inputs/tables/strategy_metrics.csv \
  --table metric_deltas=reports/demo-inputs/tables/metric_deltas.csv \
  --figure cumulative_returns=reports/demo-inputs/figures/cumulative_returns.png \
  --figure drawdowns=reports/demo-inputs/figures/drawdowns.png
```

Use `--json` to print exported artifact paths as JSON.

### run-demo-report

Create demo inputs and export a complete demo report in one workflow.

```bash
python -m regime_risk_engine run-demo-report --output-dir reports/demo-workflow
python -m regime_risk_engine run-demo-report --output-dir reports/demo-workflow --json
```

Use `--title` with `export-report` or `run-demo-report` to customize the
Markdown report title.
