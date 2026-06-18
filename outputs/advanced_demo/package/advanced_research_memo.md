# Advanced Regime-Aware Portfolio Research Memo

**Analyst:** Jimena Chinchilla

## Base Research Memo

The following section summarizes the core market research workflow output.

# Regime-Aware Portfolio Research Memo

**Analyst:** Jimena Chinchilla

## Executive Summary

The dynamic strategy was compared against the static benchmark. Overall, dynamic showed broad improvement over static, with 5 of 5 assessed metrics moving in a favorable direction. Regime-level analysis using sharpe_ratio showed that dynamic outperformed static in 3 of 3 regimes. The strongest relative regime was regime 1. The weakest relative regime was regime 0.

The analysis covers 169 return observations from 2020-01-12 to 2020-06-28. The workflow identified 3 market regimes and compared a static benchmark against a dynamic regime-aware allocation.

## Strategy Performance

The table below compares the dynamic regime-aware strategy against the static benchmark.

| Metric | Static Benchmark | Dynamic Strategy | Absolute Delta | Relative Delta |
| --- | --- | --- | --- | --- |
| Cumulative Return | 1.85% | 10.95% | 9.09% | 491.28% |
| Annualized Return | 2.77% | 16.75% | 13.98% | 504.15% |
| Annualized Volatility | 2.55% | 2.04% | -0.51% | -20.10% |
| Sharpe Ratio | 1.0878 | 8.2251 | 7.1373 | 656.10% |
| Maximum Drawdown | -10.03% | -3.02% | 7.01% | 69.88% |

## Regime-Level Findings

Regime-level analysis using sharpe_ratio showed that dynamic outperformed static in 3 of 3 regimes. The strongest relative regime was regime 1. The weakest relative regime was regime 0.

| Regime | Static Metric | Dynamic Metric | Delta | Assessment |
| --- | --- | --- | --- | --- |
| 0 | 4.1617 | 5.9235 | 1.7617 | Favorable |
| 1 | -3.1361 | 9.9761 | 13.1122 | Favorable |
| 2 | 2.2585 | 12.0715 | 9.8130 | Favorable |

## Dynamic Allocation Profile

The dynamic strategy assigns target weights by detected market regime. Weights are lagged by one period before being applied to returns to reduce look-ahead bias.

| Regime | GLD | SPY | TLT |
| --- | --- | --- | --- |
| 0 | 10.00% | 70.00% | 20.00% |
| 1 | 20.00% | 25.00% | 55.00% |
| 2 | 45.00% | 35.00% | 20.00% |

## Research Conclusion

Overall, dynamic showed broad improvement over static, with 5 of 5 assessed metrics moving in a favorable direction.

The dynamic strategy improved 5 of 5 assessed strategy metrics. The regime-level results should be reviewed to understand whether the improvement is broad-based or concentrated in specific market environments.

## Limitations

- Regime labels are model estimates, not directly observable market states.
- Historical backtests do not guarantee future performance.
- The first workflow uses a simple K-Means regime model and should be validated before production use.
- Transaction costs, liquidity constraints, taxes, and implementation frictions may reduce realized performance.
- Allocation policies should be stress-tested across different market periods before being used for real capital allocation.

## Regime Intelligence

The detected regimes were converted into interpretable market states using returns, volatility, drawdown, correlation, and asset leadership. The strongest regime was regime 0, classified as Inflation / real assets. The weakest return regime was regime 1, classified as Mixed / transition. The most severe drawdown regime was regime 1, classified as Mixed / transition.

| Regime | Label | Ann. Return | Ann. Volatility | Max Drawdown | Best Asset | Worst Asset |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Inflation / real assets | 22.17% | 0.93% | -0.35% | GLD | TLT |
| 1 | Mixed / transition | 8.17% | 1.22% | -0.67% | TLT | SPY |
| 2 | Inflation / real assets | 19.99% | 0.99% | -0.37% | GLD | SPY |

## Regime Transition Risk

Regime transition analysis estimated transition probabilities across 3 regimes using 168 observed transitions. Regime 1 was the most persistent regime with a self-transition probability of 84.09%. Regime 0 was the least persistent regime with a self-transition probability of 80.30%.

| Regime | Persistence Probability | Expected Duration | Transition Observations |
| --- | --- | --- | --- |
| 0 | 80.30% | 5.0769 | 66 |
| 1 | 84.09% | 6.2857 | 44 |
| 2 | 82.76% | 5.8000 | 58 |

## Stress-Period Analysis

Stress-period analysis compared dynamic against static across 3 stress windows. The candidate strategy protected capital in 2 of 3 periods. Average excess return across stress periods was 0.0264, and average drawdown improvement was 0.0228. The strongest stress-period result was Defensive drawdown. The weakest stress-period result was Growth expansion.

| Stress Period | Return Delta | Drawdown Delta | Volatility Delta | Assessment |
| --- | --- | --- | --- | --- |
| Growth expansion | -2.76% | 0.00% | 1.02% | Reduced drawdown but lagged return |
| Defensive drawdown | 7.03% | 6.85% | 2.06% | Protected capital |
| Inflation rotation | 3.64% | 0.00% | 0.90% | Protected capital |

## Strategy Attribution

Strategy attribution decomposed dynamic-versus-static performance into asset-level active return contributions. Total active contribution was 0.0966. The strongest positive asset contributor was GLD. Regime attribution was strongest in regime 1. Regime attribution was weakest in regime 0.

| Asset | Average Active Weight | Active Return Contribution |
| --- | --- | --- |
| GLD | 14.67% | 4.40% |
| SPY | -13.99% | 3.27% |
| TLT | -0.68% | 1.99% |

## Factor Exposure Analysis

Factor exposure analysis estimated 3 factor beta(s) for 2 strategy return series. static was most exposed to equity; dynamic was most exposed to equity. Regime-conditioned factor exposures were estimated across 3 regimes.

| Strategy | Dominant Factor | Dominant Beta |
| --- | --- | --- |
| static | equity | 0.5052 |
| dynamic | equity | 0.2772 |

## Forward Regime Scenario Simulation

Regime scenario simulation generated 1000 forward paths over a 21-period horizon using historical regime transition behavior and regime-conditioned return sampling. The simulated mean terminal return difference for dynamic versus static was 0.0100. The simulated 95% CVaR difference was 0.0144, where a higher value indicates a less severe tail outcome. The most frequently simulated regime was regime 0.

| Strategy | Mean Terminal Return | Probability of Loss | VaR 95 | CVaR 95 |
| --- | --- | --- | --- | --- |
| static | 0.31% | 32.40% | -0.90% | -1.45% |
| dynamic | 1.31% | 1.80% | 0.34% | -0.01% |

## Research Takeaway

This memo combines the base regime-aware portfolio analysis with regime interpretation, transition stability, stress-period protection, performance attribution, factor exposure diagnostics, forward scenario simulation. Together, these sections evaluate not only whether the dynamic strategy performed well, but also why it performed that way, how stable the regimes were, how the strategy behaved in stress, and what forward-looking risks remain.

## Advanced Research Limitations

- Regime labels are estimated and may change under different model settings.
- Optimized allocations can overfit if not evaluated out of sample.
- Stress-period analysis depends on selected date windows.
- Factor exposures depend on the supplied factor set.
- Scenario simulations are not forecasts; they are regime-conditioned risk simulations based on historical behavior.
