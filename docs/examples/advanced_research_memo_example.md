# Example Advanced Research Memo

This document shows an example of the type of investment research memo produced by the regime-aware portfolio risk engine.

The values below are illustrative and intended for documentation. To generate a fresh memo from the deterministic demo workflow, run:

```bash
python -m regime_risk_engine run-advanced-demo \
  --output-dir outputs/advanced_demo \
  --analyst "Jimena Chinchilla"
```

The generated memo will be written to:

```text
outputs/advanced_demo/package/advanced_research_memo.md
```

---

# Advanced Regime-Aware Portfolio Research Memo

**Analyst:** Jimena Chinchilla
**Workflow:** Advanced regime-aware portfolio research demo
**Purpose:** Evaluate whether regime-aware allocation improves portfolio risk control relative to a static benchmark.

## Executive Summary

This research memo evaluates a regime-aware allocation process that detects changing market environments and adjusts portfolio exposure dynamically.

The workflow compares a static benchmark allocation against a dynamic strategy that changes weights based on detected market regimes. The analysis includes regime intelligence, transition analysis, stress-period testing, attribution, factor exposure analysis, and scenario simulation.

The objective is not to produce live trading signals. The objective is to evaluate whether market regimes contain useful information for portfolio risk management.

## Research Question

Can changing market regimes be detected from financial time-series data, and can portfolio risk exposure be adjusted dynamically in response?

## Strategy Overview

The engine follows this research pipeline:

```text
historical prices
→ return calculation
→ feature engineering
→ regime detection
→ regime-aware allocation
→ backtesting
→ advanced risk analysis
→ investment memo export
```

The static strategy provides a benchmark allocation. The dynamic strategy applies regime-specific weights to adjust portfolio exposure across market environments.

## Regime Intelligence

Detected regimes are interpreted using return, volatility, drawdown, trend, correlation, and asset-leadership characteristics.

Example regime interpretations may include:

* **Growth / risk-on:** stronger trend, positive returns, lower stress.
* **Defensive / stress:** weaker trend, higher volatility, larger drawdowns.
* **Inflation / real-assets:** leadership from commodities or inflation-sensitive assets.
* **Mixed / transition:** unstable signals with no single dominant environment.

The goal of regime intelligence is to convert statistical clusters into economically interpretable market states.

## Regime Transition Analysis

The transition analysis estimates how regimes move from one state to another.

Important questions include:

* Which regimes are most persistent?
* Which regimes tend to follow stress periods?
* Are defensive regimes short-lived or persistent?
* Does the strategy face frequent regime switching?

Regime persistence matters because a strategy with excessive regime switching may create unnecessary turnover and unstable allocation decisions.

## Static vs Dynamic Strategy Comparison

The engine compares the static benchmark against the dynamic regime-aware strategy using risk and performance metrics such as:

* Annualized return
* Annualized volatility
* Sharpe ratio
* Maximum drawdown
* Value at Risk
* Conditional Value at Risk
* Turnover
* Stress-period performance

The key research question is whether dynamic allocation improves risk-adjusted performance or downside protection after accounting for turnover and regime uncertainty.

## Stress-Period Analysis

Stress testing evaluates performance during difficult market windows.

The analysis focuses on whether the dynamic strategy:

* Reduces drawdowns during stress regimes.
* Preserves capital better than the static benchmark.
* Avoids excessive exposure to risky assets during defensive regimes.
* Maintains reasonable participation during recovery periods.

This is one of the most important sections because regime-aware allocation is primarily valuable if it improves behavior during market stress.

## Strategy Attribution

Attribution decomposes the difference between the dynamic strategy and the static benchmark.

The analysis identifies:

* Which assets contributed most to active return.
* Which regimes contributed most to active return.
* Whether outperformance came from risk reduction, asset selection, or regime timing.
* Whether underperformance came from defensive positioning during rebounds.

This helps explain why the strategy behaved differently from the benchmark.

## Factor Exposure Analysis

Factor exposure analysis estimates relationships between strategy returns and explanatory factors.

Examples of useful factors include:

* Equity market factor
* Duration or bond factor
* Commodity factor
* Inflation-sensitive factor
* Defensive or low-volatility factor

The goal is to understand whether the strategy’s returns are explained by broad factor exposures or by regime-specific allocation behavior.

## Scenario Simulation

Scenario simulation uses regime transition behavior and regime-conditioned return samples to estimate possible forward outcomes.

The simulation helps answer:

* What range of outcomes is plausible over the next horizon?
* What is the simulated probability of loss?
* What are the simulated downside tail risks?
* Which regimes dominate adverse scenarios?

Scenario simulation does not predict the future. It provides a structured way to reason about uncertainty using observed regime behavior.

## Investment Interpretation

A strong regime-aware strategy should demonstrate several qualities:

* Better drawdown control during stress regimes.
* Sensible regime labels with economic meaning.
* Stable behavior under walk-forward validation.
* Reasonable turnover.
* Explainable attribution.
* Risk exposures that change in intuitive ways across regimes.

The strategy should not be judged only by return. For a risk engine, downside behavior, interpretability, and robustness are equally important.

## Limitations

This research engine is not a live trading system.

It does not include:

* Broker connectivity
* Live order execution
* Real-time market data feeds
* Intraday risk controls
* Tax-aware trading
* Production portfolio accounting
* Compliance review

The results should be interpreted as historical research and not as investment advice.

## Conclusion

The regime-aware portfolio risk engine provides a complete research workflow for evaluating whether market regimes can support dynamic portfolio risk management.

The project connects machine learning, portfolio analytics, stress testing, attribution, factor analysis, scenario simulation, and professional memo generation into one reproducible research pipeline.
