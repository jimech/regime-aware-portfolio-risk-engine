# Methodology

## Project objective

The goal of this project is to compare a static multi-asset portfolio against a dynamic regime-aware allocation strategy.

The core research question is:

> Can market regimes be detected from financial time series features, and can portfolio risk exposure be adjusted dynamically based on those regimes?

## Asset universe

The initial asset universe uses liquid ETF proxies across major asset classes.

The universe includes:

- U.S. equities
- International developed equities
- Emerging market equities
- U.S. Treasury bonds
- Corporate bonds
- High-yield bonds
- Inflation-protected bonds
- Gold
- Broad commodities
- Real estate
- U.S. dollar proxy
- Cash-like Treasury bill proxy

The initial universe intentionally uses ETFs instead of individual securities because ETFs provide broad exposure, cleaner data availability, and a simpler starting point for regime-aware portfolio research.

## Static benchmark portfolio

The static benchmark portfolio is defined in `configs/base.yaml`.

It acts as the baseline strategy that the dynamic regime-aware allocation will be compared against.