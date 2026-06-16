# ADR 0040: Quality checklist

## Status

Accepted

## Context

The project now has automated CI, coverage reporting, a large test suite, CLI workflows, and many analytics modules.

A visible checklist helps keep quality expectations consistent across future tickets.

## Decision

Add a project quality checklist.

The checklist documents:

- Required local checks
- Code quality expectations
- Documentation expectations
- Git hygiene
- CI expectations

The README links to the checklist and displays the GitHub Actions CI badge.

## Consequences

Contributors have a clear definition of done.

The README communicates project health through the CI badge.

Future development can follow the same repeatable quality workflow.