# ADR 0081: Advanced research package manifest

## Status

Accepted

## Context

The advanced research package exports a memo and many supporting CSV tables.

As the package grows, users and reviewers need a machine-readable way to inspect what was generated without relying only on filenames or console output.

## Decision

Add a `manifest.json` file to each advanced research package.

The manifest records:

- The generated memo filename
- The exported supporting table filenames keyed by logical table name

The advanced demo JSON CLI output also includes the manifest path.

## Consequences

The advanced research package becomes easier to inspect, validate, and consume programmatically.

This improves reproducibility and makes the package more professional for portfolio review.