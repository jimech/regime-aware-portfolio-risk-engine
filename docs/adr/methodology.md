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

## Data validation

Historical price data is validated before it is used for return calculation, feature engineering, regime detection, risk analytics, or backtesting.

The validation layer checks for:

- Missing required columns
- Empty datasets
- Invalid dates
- Duplicate date/ticker rows
- Missing adjusted close values
- Non-positive adjusted close values
- Missing expected tickers
- Unexpected tickers
- Missing observed dates by ticker

Validation issues are returned as structured objects with a severity level.

Errors represent blocking data problems that should stop the pipeline.

Warnings represent issues that may be acceptable in some research contexts but should be reviewed before interpreting results.


---

## Step 4 — Update `docs/methodology.md`

Add this section after the asset universe/static benchmark section:

```markdown
## Return calculation

The processed dataset converts adjusted close prices into daily returns.

The default return type is simple return:

```text
price_t / price_t-1 - 1


---

## Step 5 — Update `docs/methodology.md`

Add:

```markdown
## Rolling return and volatility features

The first feature engineering layer calculates rolling cumulative simple returns and annualized rolling volatility.

The default windows are:

- 21 trading days
- 63 trading days
- 126 trading days
- 252 trading days

These approximate one month, one quarter, half a year, and one trading year.

Rolling features are calculated independently by ticker and use historical observations only.