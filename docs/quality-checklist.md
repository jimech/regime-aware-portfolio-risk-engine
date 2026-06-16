
# Quality checklist

Use this checklist before committing or opening a pull request.

## Required local checks

Run the full local quality suite:

```bash
ruff format .
ruff check .
ruff format --check .
mypy src
pytest --cov=regime_risk_engine --cov-report=term-missing