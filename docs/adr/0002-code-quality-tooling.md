# ADR 0002: Code quality tooling

## Status

Accepted

## Context

The regime-aware portfolio risk engine will include multiple modules for data collection, feature engineering, regime detection, risk analytics, allocation, backtesting, validation, and reporting.

As the project grows, consistent formatting, linting, type checking, and automated tests will help prevent regressions and keep the codebase maintainable.

## Decision

Use the following development tools:

- Ruff for linting and formatting
- Pytest for automated testing
- Mypy for static type checking
- Pre-commit for running checks before commits

Ruff is selected because it is fast and can replace several common Python linting and formatting tools. Pytest is selected because it is the standard testing framework for Python projects. Mypy is selected to encourage type-safe interfaces across the project.

## Consequences

All new code should pass linting, formatting, type checking, and tests before being committed.

This creates slightly more setup work, but it improves long-term reliability and professionalism.