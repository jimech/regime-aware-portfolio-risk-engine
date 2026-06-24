# Factor Significance Example

This example explains how to interpret factor significance diagnostics in the advanced research package.

The advanced demo exports `factor_significance.csv`.

This file helps answer whether estimated factor betas are statistically distinguishable from zero.

## Why this matters

A factor beta can look economically meaningful but still be noisy.

Factor significance diagnostics help separate strong factor relationships from weak, unstable, or noisy relationships.

## Output file

The advanced research package writes `outputs/advanced_demo/package/factor_significance.csv`.

Example columns:

- strategy
- factor
- beta
- standard_error
- t_stat
- p_value
- significant

## Column meanings

| Column | Meaning |
| --- | --- |
| `strategy` | Strategy being analyzed. |
| `factor` | Factor return series used in the regression. |
| `beta` | Estimated sensitivity of strategy returns to the factor. |
| `standard_error` | Estimated uncertainty around the beta. |
| `t_stat` | Beta divided by its standard error. |
| `p_value` | Approximate probability of seeing such a result if the true beta were zero. |
| `significant` | Whether the beta passes the configured significance threshold. |

## Interpretation

A positive significant beta means the strategy had a statistically meaningful positive relationship with that factor.

A negative significant beta means the strategy had a statistically meaningful negative relationship with that factor.

A non-significant beta does not prove there is no relationship. It means the available sample does not provide strong evidence that the beta is different from zero.

## Example interpretation

The dynamic strategy showed a positive and statistically significant equity beta, suggesting that equity exposure was a meaningful driver of returns over the sample.

The defensive factor beta was small and not statistically significant, suggesting that observed defensive exposure may be noisy or unstable over the sample.

## Research use

Factor significance diagnostics support the project thesis by adding evidence quality checks to factor exposure analysis.

This feature does not forecast returns. It is a historical diagnostic used to evaluate whether estimated factor exposures are statistically meaningful.
