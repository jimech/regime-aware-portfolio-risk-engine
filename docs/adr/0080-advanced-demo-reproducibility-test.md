# ADR 0080: Advanced demo reproducibility test

## Status

Accepted

## Context

The advanced demo workflow generates a research package used as the project’s main portfolio artifact.

Because the package includes modeling, simulation, and generated outputs, deterministic reproducibility is important for testing, review, and future maintenance.

## Decision

Add a reproducibility test for the advanced demo package.

The test runs the advanced demo twice with the same random seed and compares the generated package file contents.

## Consequences

The project now protects against accidental nondeterminism in the advanced demo workflow.

This improves trust in the generated research package and makes the project easier to review.