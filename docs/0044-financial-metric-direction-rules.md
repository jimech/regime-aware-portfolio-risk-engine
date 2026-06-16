# ADR 0044: Financial metric direction rules

## Status

Accepted

## Context

The investment research layer interprets whether candidate strategy metrics improved or deteriorated relative to a benchmark.

Some metrics are straightforward:

- Higher returns are better.
- Higher Sharpe and Sortino ratios are better.
- Lower volatility and transaction costs are better.

Loss metrics require more care.

The project represents max drawdown as a negative return value. For example, `-0.18` means an 18% drawdown and `-0.25` means a 25% drawdown.

In this representation, a higher max drawdown value is better because it is less negative.

## Decision

Use explicit metric direction rules.

Higher is better for:

- Return metrics
- Risk-adjusted return metrics
- Negative loss metrics such as max drawdown, VaR, and CVaR when represented as negative return values

Lower is better for:

- Volatility
- Turnover
- Transaction costs

## Consequences

The research summary layer correctly interprets drawdown improvements.

The project avoids labeling a smaller loss as unfavorable.

Future metrics should be added to an explicit direction category before being interpreted in research summaries.