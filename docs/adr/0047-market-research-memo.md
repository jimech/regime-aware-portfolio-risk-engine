# ADR 0047: Market research memo builder

## Status

Accepted

## Context

The market research workflow can produce strategy metrics, regime-conditioned metrics, allocation outputs, and investment research summaries.

Those objects are useful for code, but a portfolio research project also needs a human-readable investment memo.

## Decision

Add a Markdown memo builder in the research layer.

The memo includes:

- Executive summary
- Strategy performance comparison
- Regime-level findings
- Dynamic allocation profile
- Research conclusion
- Limitations

The memo is generated from the market research workflow result.

## Consequences

The project can now convert quantitative workflow outputs into a professional research memo.

This supports portfolio-manager-style review and future report export workflows.