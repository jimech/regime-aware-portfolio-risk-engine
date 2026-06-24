# ADR 0086: Advanced research package loader documentation

## Status

Accepted

## Context

The project now includes a Python loader for generated advanced research packages.

Users and reviewers need documentation showing how to generate a package, load it from `manifest.json`, inspect the memo, and access exported tables as pandas DataFrames.

## Decision

Add an example document for the advanced research package loader.

The document demonstrates:

- Generating an advanced demo package
- Loading the package in Python
- Reading the memo
- Listing table names
- Accessing key exported tables

## Consequences

The package loader becomes easier to discover and use.

This prepares the project for future notebook and dashboard workflows without coupling those tools directly to package filenames.