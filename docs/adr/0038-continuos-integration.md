# ADR 0038: Continuous integration

## Status

Accepted

## Context

The project now has many modules, tests, documentation files, and CLI workflows.

Manual local checks are useful, but quality checks should also run automatically when code is pushed to GitHub or proposed through a pull request.

## Decision

Add a GitHub Actions continuous integration workflow.

The workflow runs on pushes and pull requests targeting the main branch.

The first CI workflow runs:

- Ruff formatting check
- Ruff linting
- mypy type checking
- pytest test suite

The workflow uses Python 3.12 to match the local development environment.

## Consequences

Formatting, linting, type checking, and tests are verified automatically.

Future changes are less likely to break the project silently.

Additional workflow jobs can be added later for coverage, documentation builds, package builds, and scheduled smoke tests.