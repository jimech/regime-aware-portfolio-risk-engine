import pandas as pd
import pytest

from regime_risk_engine.allocation.policy import (
    RegimeAllocationPolicyError,
    build_equal_weight_fallback,
    build_regime_target_weight_frame,
    get_target_weights_for_regime,
    validate_regime_allocation_policy,
    validate_target_weights,
)


def make_available_tickers() -> list[str]:
    return ["SPY", "TLT", "GLD"]


def make_policy() -> dict[int, dict[str, float]]:
    return {
        0: {
            "SPY": 0.70,
            "TLT": 0.20,
            "GLD": 0.10,
        },
        1: {
            "SPY": 0.30,
            "TLT": 0.50,
            "GLD": 0.20,
        },
    }


def make_fallback() -> dict[str, float]:
    return {
        "SPY": 1 / 3,
        "TLT": 1 / 3,
        "GLD": 1 / 3,
    }


def make_regime_labels() -> pd.Series:
    return pd.Series(
        [0, 0, 1, 1],
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
        name="regime",
    )


def test_validate_target_weights_success() -> None:
    weights = validate_target_weights(
        weights={
            "spy": 0.7,
            "tlt": 0.2,
            "gld": 0.1,
        },
        available_tickers=make_available_tickers(),
    )

    assert weights == {
        "SPY": 0.7,
        "TLT": 0.2,
        "GLD": 0.1,
    }


def test_validate_regime_allocation_policy_success() -> None:
    policy = validate_regime_allocation_policy(
        policy=make_policy(),
        available_tickers=make_available_tickers(),
    )

    assert set(policy) == {0, 1}
    assert policy[0]["SPY"] == pytest.approx(0.70)
    assert policy[1]["TLT"] == pytest.approx(0.50)


def test_get_target_weights_for_known_regime() -> None:
    weights = get_target_weights_for_regime(
        regime=1,
        policy=make_policy(),
        available_tickers=make_available_tickers(),
    )

    assert weights == {
        "SPY": 0.30,
        "TLT": 0.50,
        "GLD": 0.20,
    }


def test_get_target_weights_for_unknown_regime_uses_fallback() -> None:
    weights = get_target_weights_for_regime(
        regime=9,
        policy=make_policy(),
        available_tickers=make_available_tickers(),
        fallback_weights=make_fallback(),
    )

    assert weights == make_fallback()


def test_build_regime_target_weight_frame() -> None:
    labels = make_regime_labels()

    weight_frame = build_regime_target_weight_frame(
        regime_labels=labels,
        policy=make_policy(),
        available_tickers=make_available_tickers(),
    )

    assert list(weight_frame.columns) == ["GLD", "SPY", "TLT"]
    assert len(weight_frame) == 4
    assert weight_frame.loc[pd.Timestamp("2020-01-01"), "SPY"] == pytest.approx(0.70)
    assert weight_frame.loc[pd.Timestamp("2020-01-03"), "TLT"] == pytest.approx(0.50)


def test_build_regime_target_weight_frame_uses_fallback() -> None:
    labels = pd.Series(
        [0, 9],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
        name="regime",
    )

    weight_frame = build_regime_target_weight_frame(
        regime_labels=labels,
        policy=make_policy(),
        available_tickers=make_available_tickers(),
        fallback_weights=make_fallback(),
    )

    assert weight_frame.loc[pd.Timestamp("2020-01-02"), "SPY"] == pytest.approx(1 / 3)
    assert weight_frame.loc[pd.Timestamp("2020-01-02"), "TLT"] == pytest.approx(1 / 3)


def test_build_equal_weight_fallback() -> None:
    fallback = build_equal_weight_fallback(make_available_tickers())

    assert fallback == {
        "SPY": 1 / 3,
        "TLT": 1 / 3,
        "GLD": 1 / 3,
    }


def test_validate_target_weights_rejects_empty_weights() -> None:
    with pytest.raises(RegimeAllocationPolicyError, match="empty"):
        validate_target_weights({}, available_tickers=make_available_tickers())


def test_validate_target_weights_rejects_missing_ticker() -> None:
    with pytest.raises(RegimeAllocationPolicyError, match="Missing weight"):
        validate_target_weights(
            weights={
                "SPY": 0.7,
                "TLT": 0.3,
            },
            available_tickers=make_available_tickers(),
        )


def test_validate_target_weights_rejects_unknown_ticker() -> None:
    with pytest.raises(RegimeAllocationPolicyError, match="unknown ticker"):
        validate_target_weights(
            weights={
                "SPY": 0.7,
                "TLT": 0.2,
                "GLD": 0.05,
                "QQQ": 0.05,
            },
            available_tickers=make_available_tickers(),
        )


def test_validate_target_weights_rejects_negative_weight() -> None:
    with pytest.raises(RegimeAllocationPolicyError, match="non-negative"):
        validate_target_weights(
            weights={
                "SPY": 1.1,
                "TLT": -0.1,
                "GLD": 0.0,
            },
            available_tickers=make_available_tickers(),
        )


def test_validate_target_weights_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(RegimeAllocationPolicyError, match="sum to 1.0"):
        validate_target_weights(
            weights={
                "SPY": 0.7,
                "TLT": 0.7,
                "GLD": 0.1,
            },
            available_tickers=make_available_tickers(),
        )


def test_get_target_weights_rejects_unknown_regime_without_fallback() -> None:
    with pytest.raises(RegimeAllocationPolicyError, match="No allocation policy"):
        get_target_weights_for_regime(
            regime=9,
            policy=make_policy(),
            available_tickers=make_available_tickers(),
        )


def test_build_regime_target_weight_frame_rejects_non_datetime_labels() -> None:
    labels = pd.Series([0, 1], index=[0, 1], name="regime")

    with pytest.raises(RegimeAllocationPolicyError, match="DatetimeIndex"):
        build_regime_target_weight_frame(
            regime_labels=labels,
            policy=make_policy(),
            available_tickers=make_available_tickers(),
        )


def test_build_regime_target_weight_frame_rejects_missing_labels() -> None:
    labels = pd.Series(
        [0, None],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
        name="regime",
    )

    with pytest.raises(RegimeAllocationPolicyError, match="missing values"):
        build_regime_target_weight_frame(
            regime_labels=labels,
            policy=make_policy(),
            available_tickers=make_available_tickers(),
        )


def test_build_equal_weight_fallback_rejects_empty_tickers() -> None:
    with pytest.raises(RegimeAllocationPolicyError, match="empty"):
        build_equal_weight_fallback([])
