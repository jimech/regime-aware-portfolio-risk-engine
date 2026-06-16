# ADR 0008: Correlation feature design

## Status

Accepted

## Context

The regime-aware portfolio risk engine needs features that capture changing relationships between assets.

Diversification depends not only on individual asset volatility but also on how assets move together.

During crisis regimes, correlations may rise and reduce the effectiveness of diversification.

## Decision

Add rolling correlation features calculated from processed simple returns.

The first version supports:

- Rolling correlation matrices
- Average pairwise rolling correlation across all assets
- Selected rolling pairwise correlations
- Equity-bond rolling correlation

The default windows match the existing feature windows:

```text
short: 21 trading days
medium: 63 trading days
long: 126 trading days
annual: 252 trading days