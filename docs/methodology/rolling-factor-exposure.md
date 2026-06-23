# Rolling Factor Exposure Methodology

Rolling factor exposure analysis estimates how a strategy's sensitivity to risk factors changes through time.

In this project, rolling factor exposure is used to evaluate whether the dynamic regime-aware strategy changes its risk profile across market environments.

## Inputs

- A strategy return series.
- One or more factor return series.
- A rolling window length.
- A minimum observation threshold.

## Rolling regression model

For each rolling window, the engine estimates an ordinary least squares regression:

strategy_return = alpha + beta_1 * factor_1 + beta_2 * factor_2 + ... + error

- `equity_beta` measures sensitivity to the equity factor.
- `alpha` captures the return component not explained by the supplied factors.
- `r_squared` measures how much return variation is explained by the factor set.
- `residual_volatility` measures unexplained variation after factor exposures.
- `dominant_factor` identifies the factor with the largest absolute beta.

## Interpretation

The analysis is descriptive, not predictive.

## Relationship to the project thesis

Rolling factor exposure supports the project thesis by connecting allocation changes to measurable risk exposures.
It asks whether the strategy can change its market risk profile over time.
