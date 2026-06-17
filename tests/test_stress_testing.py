import pandas as pd
import pytest

from regime_risk_engine.research.stress_testing import (
    StressPeriod,
    StressTestingError,
    StressTestSummary,
    build_stress_test_summary,
)


def make_return_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    rows = []

    for index, _date in enumerate(dates):
        if index < 30:
            rows.append(
                {
                    "static": 0.002,
                    "dynamic": 0.003,
                    "regime": 0,
                }
            )
        elif index < 60:
            rows.append(
                {
                    "static": -0.006,
                    "dynamic": -0.002,
                    "regime": 1,
                }
            )
        else:
            rows.append(
                {
                    "static": -0.001,
                    "dynamic": 0.001,
                    "regime": 2,
                }
            )

    return pd.DataFrame(rows, index=dates)


def make_stress_periods() -> list[StressPeriod]:
    return [
        StressPeriod(
            name="Growth slowdown",
            start_date="2020-01-01",
            end_date="2020-01-30",
        ),
        StressPeriod(
            name="Equity drawdown",
            start_date="2020-01-31",
            end_date="2020-02-29",
        ),
        StressPeriod(
            name="Recovery chop",
            start_date="2020-03-01",
            end_date="2020-03-30",
        ),
    ]


def test_build_stress_test_summary() -> None:
    summary = build_stress_test_summary(
        return_frame=make_return_frame(),
        stress_periods=make_stress_periods(),
    )

    assert isinstance(summary, StressTestSummary)
    assert len(summary.summary_table) == 3
    assert summary.total_period_count == 3
    assert summary.protected_capital_period_count == 3
    assert summary.best_period == "Equity drawdown"
    assert "protected capital in 3 of 3 periods" in summary.narrative


def test_stress_summary_contains_expected_columns() -> None:
    summary = build_stress_test_summary(
        return_frame=make_return_frame(),
        stress_periods=make_stress_periods(),
    )

    expected_columns = {
        "period_name",
        "start_date",
        "end_date",
        "observation_count",
        "dominant_regime",
        "benchmark_cumulative_return",
        "candidate_cumulative_return",
        "return_delta",
        "benchmark_max_drawdown",
        "candidate_max_drawdown",
        "drawdown_delta",
        "benchmark_volatility",
        "candidate_volatility",
        "volatility_delta",
        "benchmark_sharpe",
        "candidate_sharpe",
        "sharpe_delta",
        "protected_capital",
        "assessment",
    }

    assert expected_columns.issubset(summary.summary_table.columns)


def test_stress_summary_identifies_dominant_regime() -> None:
    summary = build_stress_test_summary(
        return_frame=make_return_frame(),
        stress_periods=make_stress_periods(),
    )

    dominant_regimes = summary.summary_table.set_index("period_name")["dominant_regime"]

    assert dominant_regimes.loc["Growth slowdown"] == 0
    assert dominant_regimes.loc["Equity drawdown"] == 1
    assert dominant_regimes.loc["Recovery chop"] == 2


def test_stress_summary_rejects_empty_return_frame() -> None:
    with pytest.raises(StressTestingError, match="Return frame"):
        build_stress_test_summary(
            return_frame=pd.DataFrame(),
            stress_periods=make_stress_periods(),
        )


def test_stress_summary_rejects_missing_strategy_column() -> None:
    return_frame = make_return_frame().drop(columns=["dynamic"])

    with pytest.raises(StressTestingError, match="Missing strategy"):
        build_stress_test_summary(
            return_frame=return_frame,
            stress_periods=make_stress_periods(),
        )


def test_stress_summary_rejects_empty_periods() -> None:
    with pytest.raises(StressTestingError, match="At least one"):
        build_stress_test_summary(
            return_frame=make_return_frame(),
            stress_periods=[],
        )


def test_stress_summary_rejects_non_overlapping_period() -> None:
    with pytest.raises(StressTestingError, match="no overlapping"):
        build_stress_test_summary(
            return_frame=make_return_frame(),
            stress_periods=[
                StressPeriod(
                    name="No overlap",
                    start_date="1999-01-01",
                    end_date="1999-01-31",
                )
            ],
        )


def test_stress_summary_rejects_bad_period_dates() -> None:
    with pytest.raises(StressTestingError, match="start date"):
        build_stress_test_summary(
            return_frame=make_return_frame(),
            stress_periods=[
                StressPeriod(
                    name="Bad period",
                    start_date="2020-02-01",
                    end_date="2020-01-01",
                )
            ],
        )


def test_stress_summary_rejects_empty_period_name() -> None:
    with pytest.raises(StressTestingError, match="name"):
        build_stress_test_summary(
            return_frame=make_return_frame(),
            stress_periods=[
                StressPeriod(
                    name=" ",
                    start_date="2020-01-01",
                    end_date="2020-01-31",
                )
            ],
        )


def test_stress_summary_works_without_regime_column() -> None:
    return_frame = make_return_frame().drop(columns=["regime"])

    summary = build_stress_test_summary(
        return_frame=return_frame,
        stress_periods=make_stress_periods(),
    )

    assert summary.summary_table["dominant_regime"].isna().all()
