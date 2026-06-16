# ADR 0009: Feature matrix design

## Status

Accepted

## Context

Regime detection models require a clean feature matrix where each row represents one date and each column represents one model input.

Earlier feature engineering modules produce both asset-level features and date-level features.

Asset-level features include rolling returns, rolling volatility, momentum, moving-average distance, and drawdown. These are calculated by date and ticker.

Date-level features include correlation and diversification features.

## Decision

Build a date-indexed regime feature matrix.

Asset-level features are pivoted from long format into wide format using this naming convention:

```text
feature_name__TICKER