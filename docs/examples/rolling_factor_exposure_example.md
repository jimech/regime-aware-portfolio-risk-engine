# Rolling Factor Exposure Example

This example shows how to export rolling factor exposure analysis from strategy returns and factor returns.

Rolling factor exposure analysis helps answer whether a strategy’s risk profile changes over time.

For a regime-aware portfolio research engine, this is important because a dynamic strategy should ideally show behavior such as:

* Lower equity beta during stress regimes.
* Higher defensive exposure during drawdowns.
* Higher real-asset exposure during inflation-sensitive regimes.
* More stable risk exposures after optimization.

## Input files

The CLI expects two CSV files.

### Strategy returns

```text
strategy_returns.csv
```

Required columns:

```text
date,return
```

Example:

```csv
date,return
2020-01-01,0.0021
2020-01-02,-0.0014
2020-01-03,0.0032
```

### Factor returns

```text
factor_returns.csv
```

Required columns:

```text
date,<factor columns>
```

Example:

```csv
date,equity,defensive,real_asset
2020-01-01,0.0030,0.0004,0.0002
2020-01-02,-0.0020,0.0010,0.0008
2020-01-03,0.0040,-0.0003,0.0015
```

## CLI command

```bash
python -m regime_risk_engine export-rolling-factor-exposure \
  --strategy-returns path/to/strategy_returns.csv \
  --factor-returns path/to/factor_returns.csv \
  --output-dir outputs/rolling_factor_exposure \
  --window 60
```

Optional arguments include:

```text
--return-column
--date-column
--factor-column
--min-observations
--no-intercept
--json
```

Use `--factor-column` multiple times to select specific factors:

```bash
python -m regime_risk_engine export-rolling-factor-exposure \
  --strategy-returns path/to/strategy_returns.csv \
  --factor-returns path/to/factor_returns.csv \
  --output-dir outputs/rolling_factor_exposure \
  --window 60 \
  --factor-column equity \
  --factor-column defensive
```

## Output files

The command writes:

```text
rolling_factor_exposures.csv
rolling_factor_exposure_summary.csv
```

### `rolling_factor_exposures.csv`

This file contains rolling regression results by date.

Example columns:

```text
date
alpha
equity_beta
defensive_beta
real_asset_beta
r_squared
observations
residual_volatility
dominant_factor
```

### `rolling_factor_exposure_summary.csv`

This file summarizes the rolling beta history for each factor.

Example columns:

```text
factor
latest_beta
average_beta
minimum_beta
maximum_beta
beta_volatility
```

## Interpretation

Rolling factor exposure analysis helps explain whether the strategy’s behavior changed over time.

For example:

* A falling `equity_beta` during stress windows may indicate defensive risk management.
* A rising `defensive_beta` during drawdowns may indicate capital-protection behavior.
* A rising `real_asset_beta` during inflation-sensitive periods may indicate inflation-hedging exposure.
* A high `beta_volatility` may indicate unstable factor exposure.

## Research use

This feature supports the project thesis by connecting regime-aware allocation decisions to measurable changes in factor risk.

It does not predict future returns. It is a historical research tool for understanding how strategy exposures evolved over time.
