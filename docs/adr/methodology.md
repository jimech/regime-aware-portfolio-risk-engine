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

## Return calculation

The processed dataset converts adjusted close prices into daily returns.

The default return type is simple return:

```text
price_t / price_t-1 - 1


---


## Rolling return and volatility features

The first feature engineering layer calculates rolling cumulative simple returns and annualized rolling volatility.

The default windows are:

- 21 trading days
- 63 trading days
- 126 trading days
- 252 trading days

These approximate one month, one quarter, half a year, and one trading year.

Rolling features are calculated independently by ticker and use historical observations only.


---


## Momentum and trend features

The second feature engineering layer calculates momentum, moving-average distance, and drawdown.

Momentum measures cumulative simple return over a lookback window.

Moving-average distance measures whether an asset is trading above or below its recent return-index trend.

Drawdown measures the decline from the asset's running peak return index.

These features help identify market environments such as:

- Positive-trend bull markets
- Negative-momentum bear markets
- Recovery periods
- Drawdown and stress regimes


---


```markdown
## Correlation and diversification features

The third feature engineering layer calculates rolling correlation features.

The project includes:

- Rolling correlation matrices
- Average pairwise correlation across assets
- Selected pairwise correlations
- Equity-bond correlation

These features help identify diversification stress.

A rising average pairwise correlation may indicate that assets are moving together more closely, which can reduce diversification benefits during market stress.

Equity-bond correlation is tracked because the relationship between stocks and bonds can change across inflation, growth, and crisis regimes.


---

## Regime feature matrix

The feature matrix combines asset-level and date-level features into one model-ready dataset.

Asset-level features are pivoted into wide format using the naming pattern:

```text
feature_name__TICKER

## Baseline regime detection

The first regime detection model is K-Means clustering.

The model uses the date-indexed regime feature matrix and assigns each date to a numeric regime label.

K-Means is used as a baseline because it is simple, reproducible, and easy to test.

The number of regimes is configurable.

Numeric regime labels will later be interpreted using regime-specific return, volatility, drawdown, and correlation diagnostics.

## Probabilistic regime detection

The second regime detection model is a Gaussian Mixture Model.

Unlike K-Means, the Gaussian Mixture Model outputs both hard regime labels and soft regime probabilities.

This is useful for identifying periods where the market may be transitioning between regimes.

The probability output should be interpreted carefully and validated against out-of-sample behavior and regime stability diagnostics.

## Regime labeling and interpretation

Regime detection models produce numeric labels that are arbitrary by default.

The project adds a post-model interpretation layer that summarizes each regime by:

- Annualized return
- Annualized volatility
- Maximum drawdown
- Average pairwise correlation
- Observation count

These diagnostics are used to assign human-readable labels such as high-volatility stress, bull low-volatility, growth recovery, or neutral mixed.

The labeling step happens after model prediction and does not influence model training.

## Regime visualizations

The regime detection layer includes diagnostic plots.

The first set of visualizations includes:

- A timeline of detected regime labels
- Regime probability plots for probabilistic models
- Cumulative returns with regime background shading

These plots help evaluate whether model-detected regimes align with visible changes in portfolio behavior.

## Core risk metrics

The project uses reusable risk metric functions to evaluate portfolios.

The first metric layer includes:

- Cumulative return
- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Historical VaR
- Historical CVaR

VaR and CVaR are reported as positive loss values.

The default annualization factor is 252 trading days.

## Regime-specific risk analytics

After regimes are detected, the project calculates risk metrics within each regime.

The first regime-specific analytics layer includes:

- Portfolio-level risk summary by regime
- Asset-level risk summary by regime
- Equal-weight portfolio returns
- Custom weighted portfolio returns

This helps identify how return, volatility, drawdown, VaR, CVaR, and risk-adjusted performance change across market environments.

## Regime correlation and covariance analytics

The project calculates correlation and covariance matrices within each detected regime.

This helps evaluate whether diversification behaves differently across market environments.

The regime correlation layer includes:

- Correlation matrix by regime
- Annualized covariance matrix by regime
- Average pairwise correlation by regime

These diagnostics help explain why portfolio volatility and drawdown risk may change across regimes.

## Risk contribution analytics

The project calculates asset contribution to portfolio volatility.

The first risk contribution layer includes:

- Portfolio volatility from weights and covariance
- Marginal risk contribution
- Component risk contribution
- Percentage risk contribution

These analytics help identify which assets are driving portfolio risk in each regime.

Risk contribution is especially useful when comparing static portfolio weights against dynamic regime-aware allocations.

## Static benchmark allocation

The static benchmark portfolio uses fixed asset weights over the full evaluation period.

This benchmark represents a passive allocation strategy that does not respond to detected market regimes.

The dynamic regime-aware strategy will later be compared against this static baseline using the same return, risk, drawdown, and tail-risk metrics.

## Regime-aware allocation policy

The dynamic allocation strategy maps detected regimes to target asset weights.

Each regime receives a full target allocation across the available asset universe.

This rule-based policy layer is intentionally transparent. It allows the project to test whether changing portfolio exposure by regime can improve risk-adjusted performance relative to the static benchmark.

Unknown regimes may use a fallback allocation if one is provided.

## Turnover and transaction costs

The project models turnover and transaction costs for allocation changes.

One-way turnover is calculated as:

```text
0.5 * sum(abs(target_weight_i - current_weight_i))


---


## Backtest return engine

The backtesting layer converts asset returns and portfolio weights into strategy-level returns.

The engine applies portfolio weights with a default one-period lag to reduce look-ahead bias.

Gross strategy returns are calculated as:

```text
sum(asset_return_i * applied_weight_i)

## Strategy comparison framework

The backtesting layer includes a static versus dynamic comparison framework.

The comparison runs both strategies through the same return engine, transaction-cost model, and risk metric framework.

The static strategy is treated as the benchmark.

The dynamic regime-aware strategy is treated as the candidate.

Metrics are calculated on overlapping net return dates so both strategies are evaluated over the same period.

The comparison layer reports:

- Static strategy backtest results
- Dynamic strategy backtest results
- Aligned net returns
- Risk metric summary
- Candidate-minus-benchmark metric deltas