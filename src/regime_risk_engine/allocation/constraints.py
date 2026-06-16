from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


class AllocationConstraintError(ValueError):
    """Raised when allocation constraints are violated."""


@dataclass(frozen=True, slots=True)
class AssetWeightConstraint:
    """Minimum and maximum weight constraint for one asset."""

    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class AssetClassConstraint:
    """Minimum and maximum exposure constraint for one asset class."""

    min_exposure: float = 0.0
    max_exposure: float = 1.0


@dataclass(frozen=True, slots=True)
class AllocationConstraints:
    """Container for portfolio allocation constraints."""

    asset_weight_constraints: Mapping[str, AssetWeightConstraint]
    asset_class_constraints: Mapping[str, AssetClassConstraint]
    asset_class_map: Mapping[str, str]
    max_turnover: float | None = None


def validate_allocation_weights(
    weights: Mapping[str, float],
    tolerance: float = 1e-8,
) -> dict[str, float]:
    """Validate that allocation weights are non-negative and sum to one."""
    if not weights:
        raise AllocationConstraintError("Allocation weights cannot be empty")

    clean_weights = {
        str(ticker).upper().strip(): float(weight) for ticker, weight in weights.items()
    }

    if any(not ticker for ticker in clean_weights):
        raise AllocationConstraintError("Allocation tickers must be non-empty")

    if any(weight < 0 for weight in clean_weights.values()):
        raise AllocationConstraintError("Allocation weights must be non-negative")

    weight_sum = sum(clean_weights.values())

    if abs(weight_sum - 1.0) > tolerance:
        raise AllocationConstraintError("Allocation weights must sum to 1.0")

    return clean_weights


def validate_asset_weight_constraints(
    weights: Mapping[str, float],
    constraints: Mapping[str, AssetWeightConstraint],
    tolerance: float = 1e-8,
) -> None:
    """Validate asset-level min and max weight constraints."""
    clean_weights = validate_allocation_weights(weights, tolerance=tolerance)
    clean_constraints = _normalize_asset_weight_constraints(constraints)

    for ticker, constraint in clean_constraints.items():
        if ticker not in clean_weights:
            raise AllocationConstraintError(
                f"Asset weight constraint references unknown ticker: {ticker}"
            )

        weight = clean_weights[ticker]

        if weight < constraint.min_weight - tolerance:
            raise AllocationConstraintError(
                f"Ticker {ticker} weight is below minimum constraint"
            )

        if weight > constraint.max_weight + tolerance:
            raise AllocationConstraintError(
                f"Ticker {ticker} weight is above maximum constraint"
            )


def validate_asset_class_constraints(
    weights: Mapping[str, float],
    asset_class_map: Mapping[str, str],
    constraints: Mapping[str, AssetClassConstraint],
    tolerance: float = 1e-8,
) -> None:
    """Validate asset-class exposure constraints."""
    clean_weights = validate_allocation_weights(weights, tolerance=tolerance)
    clean_asset_class_map = _normalize_asset_class_map(asset_class_map)
    clean_constraints = _normalize_asset_class_constraints(constraints)

    missing_class_tickers = sorted(set(clean_weights).difference(clean_asset_class_map))

    if missing_class_tickers:
        missing = ", ".join(missing_class_tickers)
        raise AllocationConstraintError(
            f"Missing asset class mapping for ticker(s): {missing}"
        )

    exposures = calculate_asset_class_exposures(
        weights=clean_weights,
        asset_class_map=clean_asset_class_map,
    )

    for asset_class, constraint in clean_constraints.items():
        exposure = exposures.get(asset_class, 0.0)

        if exposure < constraint.min_exposure - tolerance:
            raise AllocationConstraintError(
                f"Asset class {asset_class} exposure is below minimum constraint"
            )

        if exposure > constraint.max_exposure + tolerance:
            raise AllocationConstraintError(
                f"Asset class {asset_class} exposure is above maximum constraint"
            )


def calculate_asset_class_exposures(
    weights: Mapping[str, float],
    asset_class_map: Mapping[str, str],
) -> dict[str, float]:
    """Calculate total allocation exposure by asset class."""
    clean_weights = validate_allocation_weights(weights)
    clean_asset_class_map = _normalize_asset_class_map(asset_class_map)

    exposures: dict[str, float] = {}

    for ticker, weight in clean_weights.items():
        if ticker not in clean_asset_class_map:
            raise AllocationConstraintError(
                f"Missing asset class mapping for ticker: {ticker}"
            )

        asset_class = clean_asset_class_map[ticker]
        exposures[asset_class] = exposures.get(asset_class, 0.0) + weight

    return exposures


def calculate_turnover(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
) -> float:
    """Calculate one-way portfolio turnover between two allocations."""
    clean_current = validate_allocation_weights(current_weights)
    clean_target = validate_allocation_weights(target_weights)

    if set(clean_current) != set(clean_target):
        raise AllocationConstraintError(
            "Current and target weights must contain the same tickers"
        )

    turnover = 0.5 * sum(
        abs(clean_target[ticker] - clean_current[ticker]) for ticker in clean_current
    )

    return float(turnover)


def validate_turnover_constraint(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    max_turnover: float,
    tolerance: float = 1e-8,
) -> None:
    """Validate max turnover constraint."""
    if max_turnover < 0 or max_turnover > 1:
        raise AllocationConstraintError("Maximum turnover must be between 0 and 1")

    turnover = calculate_turnover(
        current_weights=current_weights,
        target_weights=target_weights,
    )

    if turnover > max_turnover + tolerance:
        raise AllocationConstraintError("Turnover is above maximum constraint")


def validate_allocation_constraints(
    weights: Mapping[str, float],
    constraints: AllocationConstraints,
    current_weights: Mapping[str, float] | None = None,
) -> None:
    """Validate a target allocation against all configured constraints."""
    validate_allocation_weights(weights)

    validate_asset_weight_constraints(
        weights=weights,
        constraints=constraints.asset_weight_constraints,
    )
    validate_asset_class_constraints(
        weights=weights,
        asset_class_map=constraints.asset_class_map,
        constraints=constraints.asset_class_constraints,
    )

    if constraints.max_turnover is not None:
        if current_weights is None:
            raise AllocationConstraintError(
                "Current weights are required for turnover validation"
            )

        validate_turnover_constraint(
            current_weights=current_weights,
            target_weights=weights,
            max_turnover=constraints.max_turnover,
        )


def validate_weight_frame_constraints(
    weight_frame: pd.DataFrame,
    constraints: AllocationConstraints,
) -> None:
    """Validate every row in a date-indexed target weight frame."""
    if not isinstance(weight_frame.index, pd.DatetimeIndex):
        raise AllocationConstraintError("Weight frame index must be a DatetimeIndex")

    if weight_frame.empty:
        raise AllocationConstraintError("Weight frame cannot be empty")

    if weight_frame.index.has_duplicates:
        raise AllocationConstraintError("Weight frame contains duplicate dates")

    previous_weights: dict[str, float] | None = None

    for _date, row in weight_frame.sort_index().iterrows():
        weights = {str(ticker): float(weight) for ticker, weight in row.items()}

        if previous_weights is None:
            validate_allocation_constraints(
                weights=weights,
                constraints=AllocationConstraints(
                    asset_weight_constraints=constraints.asset_weight_constraints,
                    asset_class_constraints=constraints.asset_class_constraints,
                    asset_class_map=constraints.asset_class_map,
                    max_turnover=None,
                ),
            )
        else:
            validate_allocation_constraints(
                weights=weights,
                constraints=constraints,
                current_weights=previous_weights,
            )

        previous_weights = validate_allocation_weights(weights)


def _normalize_asset_weight_constraints(
    constraints: Mapping[str, AssetWeightConstraint],
) -> dict[str, AssetWeightConstraint]:
    normalized: dict[str, AssetWeightConstraint] = {}

    for ticker, constraint in constraints.items():
        clean_ticker = str(ticker).upper().strip()

        if not clean_ticker:
            raise AllocationConstraintError("Constraint tickers must be non-empty")

        if constraint.min_weight < 0 or constraint.max_weight > 1:
            raise AllocationConstraintError(
                "Asset weight constraints must be between 0 and 1"
            )

        if constraint.min_weight > constraint.max_weight:
            raise AllocationConstraintError("Asset min weight cannot exceed max weight")

        normalized[clean_ticker] = constraint

    return normalized


def _normalize_asset_class_constraints(
    constraints: Mapping[str, AssetClassConstraint],
) -> dict[str, AssetClassConstraint]:
    normalized: dict[str, AssetClassConstraint] = {}

    for asset_class, constraint in constraints.items():
        clean_asset_class = str(asset_class).lower().strip()

        if not clean_asset_class:
            raise AllocationConstraintError("Asset class names must be non-empty")

        if constraint.min_exposure < 0 or constraint.max_exposure > 1:
            raise AllocationConstraintError(
                "Asset class constraints must be between 0 and 1"
            )

        if constraint.min_exposure > constraint.max_exposure:
            raise AllocationConstraintError(
                "Asset class min exposure cannot exceed max exposure"
            )

        normalized[clean_asset_class] = constraint

    return normalized


def _normalize_asset_class_map(
    asset_class_map: Mapping[str, str],
) -> dict[str, str]:
    if not asset_class_map:
        raise AllocationConstraintError("Asset class map cannot be empty")

    normalized = {
        str(ticker).upper().strip(): str(asset_class).lower().strip()
        for ticker, asset_class in asset_class_map.items()
    }

    if any(not ticker for ticker in normalized):
        raise AllocationConstraintError("Asset class map tickers must be non-empty")

    if any(not asset_class for asset_class in normalized.values()):
        raise AllocationConstraintError("Asset class names must be non-empty")

    return normalized
