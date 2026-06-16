# ADR 0020: Backtest return engine

## Status

Accepted

## Context

The project compares static and dynamic regime-aware portfolio strategies.

A backtesting engine is needed to convert asset returns and portfolio weights into strategy-level returns.

The engine must avoid look-ahead bias. If a regime signal is observed at the end of one period, the corresponding target weights should only be applied to a later return period.

## Decision

Implement a return backtest engine that accepts:

- Long-format asset returns
- Date-indexed portfolio weights
- A configurable weight lag
- A transaction cost assumption in basis points

The default weight lag is one period.

Gross strategy return is calculated as:

```text
sum(asset_return_i * applied_weight_i)