# Regime-Aware Portfolio Risk Engine

A finance and machine learning project for detecting market regimes and comparing a static portfolio against a dynamic regime-aware allocation strategy.

## Project objective

This project investigates whether portfolio risk changes across market regimes and whether allocation decisions can be improved by adapting to those regimes.

The engine will support:

- Multi-asset market data collection
- Financial feature engineering
- Market regime detection
- Regime-specific risk analytics
- Static portfolio benchmarking
- Dynamic regime-aware allocation
- Backtesting
- Model validation
- Reporting and dashboarding
- Automated testing and CI

## Core research question

Can we detect changing market regimes and adjust portfolio risk exposure dynamically?

## Planned epics

1. Project setup
2. Data collection
3. Feature engineering
4. Regime detection
5. Regime-specific risk analytics
6. Dynamic allocation strategy
7. Backtesting engine
8. Model validation
9. Reporting and dashboard
10. Testing and CI

## Development setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run checks

Activate the virtual environment first so `ruff`, `mypy`, and `pytest` are on
your shell `PATH`:

```bash
source .venv/bin/activate
ruff check .
ruff format --check .
mypy src
pytest
pre-commit run --all-files
```


---

## Command-line usage

The project includes a command-line interface.

During local development, commands can be run with:

```bash
python -m regime_risk_engine healthcheck

## Continuous integration

The project uses GitHub Actions for continuous integration.

On pushes and pull requests to `main`, CI runs:

- Ruff formatting check
- Ruff linting
- mypy type checking
- pytest test suite

The workflow is defined in `.github/workflows/ci.yml`.