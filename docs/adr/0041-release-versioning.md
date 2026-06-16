# ADR 0041: Release and versioning process

## Status

Accepted

## Context

The project now includes analytics modules, reporting tools, CLI workflows, documentation, tests, coverage reporting, and CI.

Future releases need a repeatable process for versioning, quality checks, and documentation review.

## Decision

Adopt a lightweight release checklist and semantic versioning guidance.

The package version remains defined in `src/regime_risk_engine/__init__.py`.

Releases should be tagged using annotated Git tags, such as:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"