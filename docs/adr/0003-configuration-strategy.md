# ADR 0003: Configuration strategy

## Status

Accepted

## Context

The regime-aware portfolio risk engine will require many adjustable assumptions, including asset universe, date ranges, data paths, feature windows, regime model settings, static portfolio weights, transaction costs, and backtesting rules.

Hardcoding these values in Python modules would make the project harder to reproduce and maintain.

## Decision

Use YAML configuration files stored in the `configs/` directory.

The initial configuration file is:

```text
configs/base.yaml