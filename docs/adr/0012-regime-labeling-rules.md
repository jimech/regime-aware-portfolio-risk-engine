# ADR 0012: Regime labeling rules

## Status

Accepted

## Context

Unsupervised regime detection models produce numeric labels such as 0, 1, 2, and 3.

These numeric labels are arbitrary and do not have economic meaning until they are interpreted using regime diagnostics.

The project needs transparent rules for converting numeric clusters into readable labels for reporting and dashboarding.

## Decision

Add a post-model regime labeling layer.

The labeling layer summarizes each numeric regime using:

- Observation count
- Annualized return
- Annualized volatility
- Maximum drawdown
- Average pairwise correlation

Human-readable regime labels are assigned after model training using transparent rules.

The first label set includes:

- `high_volatility_stress`
- `bull_low_volatility`
- `growth_recovery`
- `defensive_low_return`
- `bear_or_drawdown`
- `neutral_mixed`

Labeling is diagnostic only. It does not influence model fitting or prediction.

## Consequences

Regime outputs become easier to interpret in reports and dashboards.

The numeric regime labels are preserved for reproducibility.

The first rule-based labeling approach is intentionally simple and may be refined later with more robust validation, expert review, or additional macro/market diagnostics.