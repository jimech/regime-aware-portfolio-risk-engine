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

## Backtest diagnostics

The backtesting layer includes diagnostic time series for strategy evaluation.

The first diagnostic layer includes:

- Cumulative returns
- Drawdowns
- Rolling annualized volatility
- Rolling annualized Sharpe ratio

These diagnostics help evaluate whether the dynamic regime-aware strategy behaves differently from the static benchmark through time, not only at the final summary-metric level.

## Regime-conditioned backtest evaluation

The backtesting layer evaluates strategy performance within each detected regime.

This analysis aligns strategy returns with regime labels and calculates risk metrics by regime and strategy.

The default comparison uses:

- Static strategy as the benchmark
- Dynamic regime-aware strategy as the candidate

The output includes:

- Strategy returns with regime labels
- Metric summaries by regime and strategy
- Dynamic-minus-static metric deltas by regime

This helps identify whether the dynamic strategy improves risk-adjusted performance, drawdowns, or tail risk in specific market environments.

## Time-series model validation

The validation layer uses chronological train/test splits.

The first validation utilities support:

- Expanding-window splits
- Rolling-window splits

These split methods avoid random shuffling and reduce the risk of look-ahead leakage.

Expanding-window validation uses all available past data and grows the training set over time.

Rolling-window validation uses a fixed-length training window that moves forward through time.

These splits will be used to evaluate regime model stability and strategy robustness.

## Regime stability diagnostics

The validation layer includes diagnostics for regime label stability.

The first stability diagnostics include:

- Regime distribution
- Regime transition matrix
- Transition rate
- Dominant regime share
- Label agreement across regime model outputs

Because regime detection is unsupervised, labels can be arbitrary. For that reason, the project uses label-invariant agreement metrics such as Adjusted Rand Index and normalized mutual information.

These diagnostics help evaluate whether detected regimes are stable enough to support regime-aware allocation decisions.

## Walk-forward regime validation

The validation layer includes walk-forward validation for regime detection models.

For each chronological split, the project fits a fresh model on train features and predicts regimes on both train and test features.

The validation output includes:

- Predicted train and test regime labels
- Regime counts by split
- Transition rates by split
- Dominant regime share
- Internal clustering diagnostics such as silhouette score and Calinski-Harabasz score

This reduces look-ahead leakage and helps evaluate whether regime model outputs are stable across time.

## Regime model selection

The validation layer includes a model selection summary for comparing candidate regime detection models.

The first model selection layer summarizes walk-forward validation results across models using:

- Regime counts
- Transition rates
- Dominant regime shares
- Silhouette scores
- Calinski-Harabasz scores

Models can be ranked by a selected metric.

The default ranking metric is test silhouette score, but final model choice should also consider interpretability, stability, regime-specific risk behavior, and backtest performance.

## Reporting table layer

The reporting layer converts analytical outputs into clean report-ready tables.

The first reporting layer supports:

- Strategy metric tables
- Metric delta tables
- Regime-conditioned metric tables
- Regime model ranking tables

These tables preserve machine-readable metric names and add readable labels for presentation.

This creates a clean handoff from analytics modules to notebooks, dashboards, and final reporting.


---

## Report exports

The reporting layer includes utilities for exporting report artifacts.

The first export layer supports:

- CSV table exports
- PNG figure exports
- Markdown report index generation
- JSON manifest generation

Exports are written into a structured output directory with separate table and figure folders.

This makes analytical results easier to review, share, and reproduce.

## Report assembly

The reporting layer includes a high-level report assembly workflow.

The assembly layer combines:

- Strategy comparison results
- Backtest diagnostics
- Regime-conditioned strategy evaluation
- Regime model selection outputs

It produces report-ready tables and figures, and can optionally export them into a structured report directory.

This creates a reproducible path from analytics outputs to final project reporting.

## Command-line interface

The project includes a lightweight command-line interface.

The first CLI layer supports:

- Version checks
- Basic project healthchecks
- Optional output directory validation and creation

This provides a terminal entry point for the project and creates a foundation for future workflow commands such as data download, feature generation, model validation, backtesting, and report export.


---

## CLI report export

The command-line interface includes a report export command.

The command accepts named CSV tables and optional PNG figures, then writes a structured report folder containing:

- Exported tables
- Exported figures
- Markdown report index
- JSON manifest

This allows report artifacts to be generated from the terminal without writing Python code.


---


## CLI config inspection

The command-line interface includes a config inspection command.

The command reads a YAML config file and reports:

- Top-level config keys
- Detected tickers
- Detected start and end dates
- Detected data directory
- Detected report directory

This helps users verify project configuration before running longer workflows.


---


## CLI demo report workflow

The command-line interface includes a one-step demo report workflow.

The command creates demo input tables and figures, exports them into a report folder, and generates a Markdown index and JSON manifest.

This provides a quick smoke test for the reporting workflow and helps users verify that the CLI, reporting export, and demo generation layers work together.