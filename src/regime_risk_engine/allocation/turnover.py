from collections.abc import Mapping

import pandas as pd


class TurnoverModelError(ValueError):
    """Raised when turnover or transaction costs cannot be calculated."""


def calculate_one_way_turnover(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
) -> float:
    """Calculate one-way turnover between two portfolio allocations.

    Turnover is:

    0.5 * sum(abs(target_weight_i - current_weight_i))
    """
    clean_current = _validate_weights(current_weights)
    clean_target = _validate_weights(target_weights)

    if set(clean_current) != set(clean_target):
        raise TurnoverModelError(
            "Current and target weights must contain the same tickers"
        )

    turnover = 0.5 * sum(
        abs(clean_target[ticker] - clean_current[ticker]) for ticker in clean_current
    )

    return float(turnover)


def calculate_turnover_series(weight_frame: pd.DataFrame) -> pd.Series:
    """Calculate one-way turnover for each date in a weight frame.

    The first date has turnover of 0.0 because there is no prior allocation.
    """
    clean_weight_frame = _validate_weight_frame(weight_frame)

    turnover_values: list[float] = []

    previous_weights: dict[str, float] | None = None

    for _, row in clean_weight_frame.iterrows():
        current_weights = {str(ticker): float(weight) for ticker, weight in row.items()}

        if previous_weights is None:
            turnover_values.append(0.0)
        else:
            turnover_values.append(
                calculate_one_way_turnover(
                    current_weights=previous_weights,
                    target_weights=current_weights,
                )
            )

        previous_weights = current_weights

    turnover = pd.Series(
        turnover_values,
        index=clean_weight_frame.index,
        name="turnover",
    )

    return turnover


def calculate_transaction_costs(
    turnover: pd.Series,
    transaction_cost_bps: float,
) -> pd.Series:
    """Calculate transaction costs from turnover and basis-point cost.

    Transaction cost is expressed as a return drag:

    turnover * transaction_cost_bps / 10_000
    """
    clean_turnover = _validate_turnover_series(turnover)
    _validate_transaction_cost_bps(transaction_cost_bps)

    costs = clean_turnover * (transaction_cost_bps / 10_000.0)
    costs.name = "transaction_cost"

    return costs


def calculate_net_returns_after_costs(
    gross_returns: pd.Series,
    transaction_costs: pd.Series,
) -> pd.Series:
    """Subtract transaction costs from gross strategy returns."""
    clean_returns = _validate_return_series(gross_returns)
    clean_costs = _validate_cost_series(transaction_costs)

    overlapping_dates = clean_returns.index.intersection(clean_costs.index)

    if overlapping_dates.empty:
        raise TurnoverModelError(
            "Gross returns and transaction costs have no overlapping dates"
        )

    aligned_returns = clean_returns.loc[overlapping_dates].sort_index()
    aligned_costs = clean_costs.loc[overlapping_dates].sort_index()

    net_returns = aligned_returns - aligned_costs
    net_returns.name = "net_return"

    return net_returns


def build_transaction_cost_summary(
    weight_frame: pd.DataFrame,
    gross_returns: pd.Series,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Build turnover, transaction cost, gross return, and net return summary."""
    turnover = calculate_turnover_series(weight_frame)
    costs = calculate_transaction_costs(
        turnover=turnover,
        transaction_cost_bps=transaction_cost_bps,
    )
    net_returns = calculate_net_returns_after_costs(
        gross_returns=gross_returns,
        transaction_costs=costs,
    )

    overlapping_dates = net_returns.index
    clean_returns = _validate_return_series(gross_returns).loc[overlapping_dates]

    summary = pd.DataFrame(
        {
            "gross_return": clean_returns,
            "turnover": turnover.loc[overlapping_dates],
            "transaction_cost": costs.loc[overlapping_dates],
            "net_return": net_returns,
        },
        index=overlapping_dates,
    )
    summary.index.name = "date"

    return summary


def _validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights:
        raise TurnoverModelError("Portfolio weights cannot be empty")

    clean_weights = {
        str(ticker).upper().strip(): float(weight) for ticker, weight in weights.items()
    }

    if any(not ticker for ticker in clean_weights):
        raise TurnoverModelError("Portfolio tickers must be non-empty")

    if any(weight < 0 for weight in clean_weights.values()):
        raise TurnoverModelError("Portfolio weights must be non-negative")

    weight_sum = sum(clean_weights.values())

    if abs(weight_sum - 1.0) > 1e-8:
        raise TurnoverModelError("Portfolio weights must sum to 1.0")

    return clean_weights


def _validate_weight_frame(weight_frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(weight_frame.index, pd.DatetimeIndex):
        raise TurnoverModelError("Weight frame index must be a DatetimeIndex")

    if weight_frame.empty:
        raise TurnoverModelError("Weight frame cannot be empty")

    if weight_frame.index.has_duplicates:
        raise TurnoverModelError("Weight frame contains duplicate dates")

    if weight_frame.columns.has_duplicates:
        raise TurnoverModelError("Weight frame contains duplicate tickers")

    if weight_frame.isna().any().any():
        raise TurnoverModelError("Weight frame contains missing values")

    clean_weight_frame = weight_frame.copy()
    clean_weight_frame.columns = [
        str(ticker).upper().strip() for ticker in clean_weight_frame.columns
    ]

    if any(not ticker for ticker in clean_weight_frame.columns):
        raise TurnoverModelError("Weight frame tickers must be non-empty")

    clean_weight_frame = clean_weight_frame.astype(float).sort_index()

    for _, row in clean_weight_frame.iterrows():
        _validate_weights(
            {str(ticker): float(weight) for ticker, weight in row.items()}
        )

    return clean_weight_frame


def _validate_turnover_series(turnover: pd.Series) -> pd.Series:
    if not isinstance(turnover.index, pd.DatetimeIndex):
        raise TurnoverModelError("Turnover index must be a DatetimeIndex")

    if turnover.empty:
        raise TurnoverModelError("Turnover series cannot be empty")

    if turnover.index.has_duplicates:
        raise TurnoverModelError("Turnover series contains duplicate dates")

    clean_turnover = pd.to_numeric(turnover, errors="coerce")

    if clean_turnover.isna().any():
        raise TurnoverModelError("Turnover series contains missing values")

    if (clean_turnover < 0).any():
        raise TurnoverModelError("Turnover cannot be negative")

    return pd.Series(
        clean_turnover,
        index=turnover.index,
        name=turnover.name,
    ).sort_index()


def _validate_transaction_cost_bps(transaction_cost_bps: float) -> None:
    if transaction_cost_bps < 0:
        raise TurnoverModelError("Transaction cost basis points cannot be negative")


def _validate_return_series(returns: pd.Series) -> pd.Series:
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TurnoverModelError("Return series index must be a DatetimeIndex")

    if returns.empty:
        raise TurnoverModelError("Return series cannot be empty")

    if returns.index.has_duplicates:
        raise TurnoverModelError("Return series contains duplicate dates")

    clean_returns = pd.to_numeric(returns, errors="coerce")

    if clean_returns.isna().any():
        raise TurnoverModelError("Return series contains missing values")

    return pd.Series(
        clean_returns,
        index=returns.index,
        name=returns.name,
    ).sort_index()


def _validate_cost_series(costs: pd.Series) -> pd.Series:
    if not isinstance(costs.index, pd.DatetimeIndex):
        raise TurnoverModelError("Transaction cost index must be a DatetimeIndex")

    if costs.empty:
        raise TurnoverModelError("Transaction cost series cannot be empty")

    if costs.index.has_duplicates:
        raise TurnoverModelError("Transaction cost series contains duplicate dates")

    clean_costs = pd.to_numeric(costs, errors="coerce")

    if clean_costs.isna().any():
        raise TurnoverModelError("Transaction cost series contains missing values")

    if (clean_costs < 0).any():
        raise TurnoverModelError("Transaction costs cannot be negative")

    return pd.Series(
        clean_costs,
        index=costs.index,
        name=costs.name,
    ).sort_index()
