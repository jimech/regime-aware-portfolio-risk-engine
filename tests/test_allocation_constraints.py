import pandas as pd
import pytest

from regime_risk_engine.allocation.constraints import (
    AllocationConstraintError,
    AllocationConstraints,
    AssetClassConstraint,
    AssetWeightConstraint,
    calculate_asset_class_exposures,
    calculate_turnover,
    validate_allocation_constraints,
    validate_allocation_weights,
    validate_asset_class_constraints,
    validate_asset_weight_constraints,
    validate_turnover_constraint,
    validate_weight_frame_constraints,
)


def make_weights() -> dict[str, float]:
    return {
        "SPY": 0.50,
        "TLT": 0.30,
        "GLD": 0.20,
    }


def make_current_weights() -> dict[str, float]:
    return {
        "SPY": 0.40,
        "TLT": 0.40,
        "GLD": 0.20,
    }


def make_asset_class_map() -> dict[str, str]:
    return {
        "SPY": "equity",
        "TLT": "fixed_income",
        "GLD": "commodity",
    }


def make_constraints(max_turnover: float | None = None) -> AllocationConstraints:
    return AllocationConstraints(
        asset_weight_constraints={
            "SPY": AssetWeightConstraint(min_weight=0.20, max_weight=0.70),
            "TLT": AssetWeightConstraint(min_weight=0.10, max_weight=0.60),
            "GLD": AssetWeightConstraint(min_weight=0.00, max_weight=0.30),
        },
        asset_class_constraints={
            "equity": AssetClassConstraint(min_exposure=0.20, max_exposure=0.70),
            "fixed_income": AssetClassConstraint(
                min_exposure=0.10,
                max_exposure=0.60,
            ),
            "commodity": AssetClassConstraint(min_exposure=0.00, max_exposure=0.30),
        },
        asset_class_map=make_asset_class_map(),
        max_turnover=max_turnover,
    )


def test_validate_allocation_weights_success() -> None:
    weights = validate_allocation_weights(make_weights())

    assert weights == make_weights()


def test_calculate_asset_class_exposures() -> None:
    exposures = calculate_asset_class_exposures(
        weights=make_weights(),
        asset_class_map=make_asset_class_map(),
    )

    assert exposures == {
        "equity": 0.50,
        "fixed_income": 0.30,
        "commodity": 0.20,
    }


def test_validate_asset_weight_constraints_success() -> None:
    validate_asset_weight_constraints(
        weights=make_weights(),
        constraints=make_constraints().asset_weight_constraints,
    )


def test_validate_asset_class_constraints_success() -> None:
    validate_asset_class_constraints(
        weights=make_weights(),
        asset_class_map=make_asset_class_map(),
        constraints=make_constraints().asset_class_constraints,
    )


def test_calculate_turnover() -> None:
    turnover = calculate_turnover(
        current_weights=make_current_weights(),
        target_weights=make_weights(),
    )

    assert turnover == pytest.approx(0.10)


def test_validate_turnover_constraint_success() -> None:
    validate_turnover_constraint(
        current_weights=make_current_weights(),
        target_weights=make_weights(),
        max_turnover=0.20,
    )


def test_validate_allocation_constraints_success() -> None:
    validate_allocation_constraints(
        weights=make_weights(),
        constraints=make_constraints(),
    )


def test_validate_allocation_constraints_with_turnover_success() -> None:
    validate_allocation_constraints(
        weights=make_weights(),
        constraints=make_constraints(max_turnover=0.20),
        current_weights=make_current_weights(),
    )


def test_validate_weight_frame_constraints_success() -> None:
    weight_frame = pd.DataFrame(
        [
            {"SPY": 0.40, "TLT": 0.40, "GLD": 0.20},
            {"SPY": 0.50, "TLT": 0.30, "GLD": 0.20},
        ],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    validate_weight_frame_constraints(
        weight_frame=weight_frame,
        constraints=make_constraints(max_turnover=0.20),
    )


def test_validate_allocation_weights_rejects_empty_weights() -> None:
    with pytest.raises(AllocationConstraintError, match="empty"):
        validate_allocation_weights({})


def test_validate_allocation_weights_rejects_negative_weight() -> None:
    with pytest.raises(AllocationConstraintError, match="non-negative"):
        validate_allocation_weights(
            {
                "SPY": 1.1,
                "TLT": -0.1,
            }
        )


def test_validate_allocation_weights_rejects_sum_not_one() -> None:
    with pytest.raises(AllocationConstraintError, match="sum to 1.0"):
        validate_allocation_weights(
            {
                "SPY": 0.7,
                "TLT": 0.7,
            }
        )


def test_validate_asset_weight_constraints_rejects_below_minimum() -> None:
    with pytest.raises(AllocationConstraintError, match="below minimum"):
        validate_asset_weight_constraints(
            weights={
                "SPY": 0.10,
                "TLT": 0.60,
                "GLD": 0.30,
            },
            constraints=make_constraints().asset_weight_constraints,
        )


def test_validate_asset_weight_constraints_rejects_above_maximum() -> None:
    with pytest.raises(AllocationConstraintError, match="above maximum"):
        validate_asset_weight_constraints(
            weights={
                "SPY": 0.80,
                "TLT": 0.10,
                "GLD": 0.10,
            },
            constraints=make_constraints().asset_weight_constraints,
        )


def test_validate_asset_class_constraints_rejects_missing_mapping() -> None:
    with pytest.raises(AllocationConstraintError, match="Missing asset class"):
        validate_asset_class_constraints(
            weights=make_weights(),
            asset_class_map={
                "SPY": "equity",
                "TLT": "fixed_income",
            },
            constraints=make_constraints().asset_class_constraints,
        )


def test_validate_asset_class_constraints_rejects_above_maximum() -> None:
    with pytest.raises(AllocationConstraintError, match="above maximum"):
        validate_asset_class_constraints(
            weights={
                "SPY": 0.80,
                "TLT": 0.10,
                "GLD": 0.10,
            },
            asset_class_map=make_asset_class_map(),
            constraints=make_constraints().asset_class_constraints,
        )


def test_validate_turnover_constraint_rejects_high_turnover() -> None:
    with pytest.raises(AllocationConstraintError, match="Turnover"):
        validate_turnover_constraint(
            current_weights={
                "SPY": 1.0,
                "TLT": 0.0,
                "GLD": 0.0,
            },
            target_weights={
                "SPY": 0.0,
                "TLT": 1.0,
                "GLD": 0.0,
            },
            max_turnover=0.20,
        )


def test_validate_turnover_constraint_rejects_invalid_limit() -> None:
    with pytest.raises(AllocationConstraintError, match="between 0 and 1"):
        validate_turnover_constraint(
            current_weights=make_current_weights(),
            target_weights=make_weights(),
            max_turnover=1.5,
        )


def test_validate_allocation_constraints_requires_current_weights_for_turnover() -> (
    None
):
    with pytest.raises(AllocationConstraintError, match="Current weights"):
        validate_allocation_constraints(
            weights=make_weights(),
            constraints=make_constraints(max_turnover=0.20),
        )


def test_validate_weight_frame_constraints_rejects_non_datetime_index() -> None:
    weight_frame = pd.DataFrame(
        [{"SPY": 0.50, "TLT": 0.30, "GLD": 0.20}],
        index=[0],
    )

    with pytest.raises(AllocationConstraintError, match="DatetimeIndex"):
        validate_weight_frame_constraints(
            weight_frame=weight_frame,
            constraints=make_constraints(),
        )
