# ADR 0089: Streamlit dashboard MVP

## Status

Accepted

## Context

The project now exports advanced research packages with a manifest, package loader, and dashboard-ready summary helper.

A lightweight dashboard can help reviewers inspect the generated memo and supporting tables without writing Python code.

## Decision

Add a minimal Streamlit dashboard.

The dashboard loads an existing advanced research package from `manifest.json`, summarizes available diagnostics, renders the memo, and previews exported tables.

The dashboard does not run the full modeling workflow live. It only reads a generated package, keeping the UI simple and reproducible.

## Consequences

The project gains an interactive review surface for generated research packages.

Future work can add charts, richer table views, and package generation controls.