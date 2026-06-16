import pandas as pd
import pytest

from regime_risk_engine.allocation.turnover import (
    TurnoverModelError,
    build_transaction_cost_summary,
    calculate_net_returns_after_costs,
    calculate_one_way_turnover,
    calculate_transaction_costs,
    calculate_turnover_series,
)


def make_current_weights() -> dict[str, float]:
    return {
        "SPY": 0.60,
        "TLT": 0.30,
        "GLD": 0.10,
    }


def make_target_weights() -> dict[str, float]:
    return {
        "SPY": 0.40,
        "TLT": 0.40,
        "GLD": 0.20,
    }


def make_weight_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"SPY": 0.60, "TLT": 0.30, "GLD": 0.10},
            {"SPY": 0.40, "TLT": 0.40, "GLD": 0.20},
            {"SPY": 0.50, "TLT": 0.30, "GLD": 0.20},
        ],
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )


def make_gross_returns() -> pd.Series:
    return pd.Series(
        [0.01, 0.02, -0.01],
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
        name="gross_return",
    )


def test_calculate_one_way_turnover() -> None:
    turnover = calculate_one_way_turnover(
        current_weights=make_current_weights(),
        target_weights=make_target_weights(),
    )

    expected = 0.5 * (0.20 + 0.10 + 0.10)

    assert turnover == pytest.approx(expected)


def test_calculate_turnover_series() -> None:
    weight_frame = make_weight_frame()

    turnover = calculate_turnover_series(weight_frame)

    assert turnover.name == "turnover"
    assert turnover.iloc[0] == pytest.approx(0.0)
    assert turnover.iloc[1] == pytest.approx(0.20)
    assert turnover.iloc[2] == pytest.approx(0.10)


def test_calculate_transaction_costs() -> None:
    turnover = calculate_turnover_series(make_weight_frame())

    costs = calculate_transaction_costs(
        turnover=turnover,
        transaction_cost_bps=10.0,
    )

    assert costs.name == "transaction_cost"
    assert costs.iloc[0] == pytest.approx(0.0)
    assert costs.iloc[1] == pytest.approx(0.0002)


def test_calculate_net_returns_after_costs() -> None:
    gross_returns = make_gross_returns()
    costs = pd.Series(
        [0.0, 0.0002, 0.0001],
        index=gross_returns.index,
        name="transaction_cost",
    )

    net_returns = calculate_net_returns_after_costs(
        gross_returns=gross_returns,
        transaction_costs=costs,
    )

    assert net_returns.name == "net_return"
    assert net_returns.iloc[0] == pytest.approx(0.01)
    assert net_returns.iloc[1] == pytest.approx(0.0198)
    assert net_returns.iloc[2] == pytest.approx(-0.0101)


def test_build_transaction_cost_summary() -> None:
    summary = build_transaction_cost_summary(
        weight_frame=make_weight_frame(),
        gross_returns=make_gross_returns(),
        transaction_cost_bps=10.0,
    )

    assert list(summary.columns) == [
        "gross_return",
        "turnover",
        "transaction_cost",
        "net_return",
    ]
    assert len(summary) == 3
    assert summary.iloc[1]["turnover"] == pytest.approx(0.20)
    assert summary.iloc[1]["transaction_cost"] == pytest.approx(0.0002)
    assert summary.iloc[1]["net_return"] == pytest.approx(0.0198)


def test_calculate_one_way_turnover_rejects_mismatched_tickers() -> None:
    with pytest.raises(TurnoverModelError, match="same tickers"):
        calculate_one_way_turnover(
            current_weights={
                "SPY": 1.0,
            },
            target_weights={
                "TLT": 1.0,
            },
        )


def test_calculate_one_way_turnover_rejects_negative_weights() -> None:
    with pytest.raises(TurnoverModelError, match="non-negative"):
        calculate_one_way_turnover(
            current_weights={
                "SPY": 1.1,
                "TLT": -0.1,
            },
            target_weights={
                "SPY": 0.5,
                "TLT": 0.5,
            },
        )


def test_calculate_turnover_series_rejects_non_datetime_index() -> None:
    weight_frame = pd.DataFrame(
        [
            {"SPY": 0.6, "TLT": 0.4},
        ],
        index=[0],
    )

    with pytest.raises(TurnoverModelError, match="DatetimeIndex"):
        calculate_turnover_series(weight_frame)


def test_calculate_turnover_series_rejects_missing_values() -> None:
    weight_frame = pd.DataFrame(
        [
            {"SPY": 0.6, "TLT": 0.4},
            {"SPY": None, "TLT": 1.0},
        ],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    with pytest.raises(TurnoverModelError, match="missing values"):
        calculate_turnover_series(weight_frame)


def test_calculate_transaction_costs_rejects_negative_bps() -> None:
    turnover = calculate_turnover_series(make_weight_frame())

    with pytest.raises(TurnoverModelError, match="basis points"):
        calculate_transaction_costs(
            turnover=turnover,
            transaction_cost_bps=-1.0,
        )


def test_calculate_transaction_costs_rejects_negative_turnover() -> None:
    turnover = pd.Series(
        [0.0, -0.1],
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
        name="turnover",
    )

    with pytest.raises(TurnoverModelError, match="negative"):
        calculate_transaction_costs(
            turnover=turnover,
            transaction_cost_bps=10.0,
        )


def test_calculate_net_returns_rejects_no_overlapping_dates() -> None:
    gross_returns = pd.Series(
        [0.01],
        index=pd.to_datetime(["2020-01-01"]),
        name="gross_return",
    )
    costs = pd.Series(
        [0.001],
        index=pd.to_datetime(["2030-01-01"]),
        name="transaction_cost",
    )

    with pytest.raises(TurnoverModelError, match="no overlapping dates"):
        calculate_net_returns_after_costs(
            gross_returns=gross_returns,
            transaction_costs=costs,
        )


def test_build_transaction_cost_summary_aligns_on_dates() -> None:
    weight_frame = make_weight_frame()
    gross_returns = pd.Series(
        [0.02, -0.01],
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        name="gross_return",
    )

    summary = build_transaction_cost_summary(
        weight_frame=weight_frame,
        gross_returns=gross_returns,
        transaction_cost_bps=10.0,
    )

    assert list(summary.index) == list(gross_returns.index)
    assert len(summary) == 2
