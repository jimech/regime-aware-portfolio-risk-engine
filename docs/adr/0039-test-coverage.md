# ADR 0039: Test coverage reporting

## Status

Accepted

## Context

The project has a large test suite covering data, features, regimes, risk analytics, allocation, backtesting, validation, reporting, and CLI workflows.

As the project grows, test count alone is not enough. The project should also measure how much production code is exercised by tests.

## Decision

Add test coverage reporting using pytest-cov.

Coverage is measured for the `regime_risk_engine` package.

The initial minimum coverage threshold is 80%.

CI runs tests with coverage reporting.

## Consequences

The project gains a quantitative signal for test completeness.

Pull requests and future changes are less likely to reduce coverage accidentally.

The coverage threshold can be raised later as the project matures.