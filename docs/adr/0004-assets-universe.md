# ADR 0004: Asset universe

## Status

Accepted

## Context

The regime-aware portfolio risk engine needs a multi-asset universe that can behave differently across market environments.

The project should support analysis of bull markets, bear markets, inflation shocks, risk-off crises, and recovery periods.

A narrow equity-only universe would not provide enough diversification behavior to analyze regime-specific risk.

## Decision

Start with a liquid ETF-based universe containing 15 to 25 assets.

The initial universe includes exposures to:

- U.S. equities
- International developed equities
- Emerging market equities
- Treasury bonds
- Corporate credit
- High-yield credit
- Inflation-protected bonds
- Gold
- Broad commodities
- Real estate
- U.S. dollar proxy
- Cash-like Treasury bills

The project will use ETF proxies instead of individual securities in the first version.

## Consequences

ETF proxies simplify data collection and make the project easier to reproduce.

The universe is broad enough to study changing volatility, drawdowns, correlations, and risk contributions across regimes.

However, ETF histories may have different inception dates, liquidity profiles, fees, and tracking errors. These limitations should be considered when interpreting backtest results.