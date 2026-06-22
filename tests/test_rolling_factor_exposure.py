from pathlib import Path

import pandas as pd
import pytest

from regime_risk_engine.research.rolling_factor_exposure import (
    RollingFactorExposureError,
    estimate_rolling_factor_exposures,
    read_rolling_factor_exposure_inputs,
    summarize_rolling_factor_exposures,
    write_rolling_factor_exposures_csv,
)


def test_estimate_rolling_factor_exposures_recovers_known_beta() -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    equity = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.03, 0.01]
    strategy = [0.001 + 2.0 * value for value in equity]

    result = estimate_rolling_factor_exposures(
        strategy_returns=pd.DataFrame({"date": dates, "return": strategy}),
        factor_returns=pd.DataFrame({"date": dates, "equity": equity}),
        window=5,
    )

    assert not result.exposures.empty
    assert result.factor_columns == ("equity",)
    assert result.exposures["equity_beta"].iloc[-1] == pytest.approx(2.0)
    assert result.exposures["alpha"].iloc[-1] == pytest.approx(0.001)
    assert result.exposures["dominant_factor"].iloc[-1] == "equity"


def test_estimate_rolling_factor_exposures_supports_multiple_factors() -> None:
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    equity = [
        0.01,
        0.02,
        -0.01,
        0.03,
        -0.02,
        0.01,
        0.02,
        -0.01,
        0.03,
        0.01,
        0.02,
        -0.01,
    ]
    rates = [
        -0.01,
        0.00,
        0.01,
        -0.02,
        0.01,
        0.00,
        -0.01,
        0.02,
        -0.01,
        0.00,
        0.01,
        -0.02,
    ]
    strategy = [
        0.0005 + 1.5 * equity_value - 0.5 * rates_value
        for equity_value, rates_value in zip(equity, rates, strict=True)
    ]

    result = estimate_rolling_factor_exposures(
        strategy_returns=pd.DataFrame({"date": dates, "return": strategy}),
        factor_returns=pd.DataFrame(
            {
                "date": dates,
                "equity": equity,
                "rates": rates,
            }
        ),
        window=8,
    )

    latest = result.latest_exposures()

    assert latest["equity_beta"] == pytest.approx(1.5)
    assert latest["rates_beta"] == pytest.approx(-0.5)


def test_estimate_rolling_factor_exposures_supports_no_intercept() -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    equity = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01]
    strategy = [3.0 * value for value in equity]

    result = estimate_rolling_factor_exposures(
        strategy_returns=pd.DataFrame({"date": dates, "return": strategy}),
        factor_returns=pd.DataFrame({"date": dates, "equity": equity}),
        window=4,
        include_intercept=False,
    )

    assert result.exposures["alpha"].iloc[-1] == pytest.approx(0.0)
    assert result.exposures["equity_beta"].iloc[-1] == pytest.approx(3.0)


def test_estimate_rolling_factor_exposures_handles_missing_values() -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    strategy = [0.01, 0.02, None, 0.03, 0.01, 0.02, 0.03, 0.04]
    equity = [0.01, 0.02, 0.03, None, 0.01, 0.02, 0.03, 0.04]

    result = estimate_rolling_factor_exposures(
        strategy_returns=pd.DataFrame({"date": dates, "return": strategy}),
        factor_returns=pd.DataFrame({"date": dates, "equity": equity}),
        window=5,
        min_observations=3,
    )

    assert not result.exposures.empty
    assert result.exposures["observations"].min() >= 3


def test_summarize_rolling_factor_exposures_returns_factor_summary() -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    equity = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.03, 0.01]
    strategy = [0.001 + 2.0 * value for value in equity]

    result = estimate_rolling_factor_exposures(
        strategy_returns=pd.DataFrame({"date": dates, "return": strategy}),
        factor_returns=pd.DataFrame({"date": dates, "equity": equity}),
        window=5,
    )
    summary = summarize_rolling_factor_exposures(result)

    assert list(summary.columns) == [
        "factor",
        "latest_beta",
        "average_beta",
        "minimum_beta",
        "maximum_beta",
        "beta_volatility",
    ]
    assert summary.loc[0, "factor"] == "equity"
    assert summary.loc[0, "latest_beta"] == pytest.approx(2.0)


def test_read_and_write_rolling_factor_exposures_csv(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    equity = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01]
    strategy = [2.0 * value for value in equity]

    strategy_path = tmp_path / "strategy.csv"
    factor_path = tmp_path / "factors.csv"
    output_path = tmp_path / "nested" / "rolling_exposures.csv"

    pd.DataFrame({"date": dates, "return": strategy}).to_csv(strategy_path, index=False)
    pd.DataFrame({"date": dates, "equity": equity}).to_csv(factor_path, index=False)

    result = read_rolling_factor_exposure_inputs(
        strategy_returns_path=strategy_path,
        factor_returns_path=factor_path,
        window=4,
        include_intercept=False,
    )

    written_path = write_rolling_factor_exposures_csv(output_path, result)

    assert written_path == output_path
    assert output_path.exists()

    written = pd.read_csv(output_path)

    assert "equity_beta" in written.columns


def test_estimate_rolling_factor_exposures_rejects_invalid_window() -> None:
    with pytest.raises(RollingFactorExposureError, match="window"):
        estimate_rolling_factor_exposures(
            strategy_returns=pd.DataFrame({"date": ["2020-01-01"], "return": [0.01]}),
            factor_returns=pd.DataFrame({"date": ["2020-01-01"], "equity": [0.01]}),
            window=1,
        )


def test_estimate_rolling_factor_exposures_rejects_missing_columns() -> None:
    with pytest.raises(RollingFactorExposureError, match="missing column"):
        estimate_rolling_factor_exposures(
            strategy_returns=pd.DataFrame({"date": ["2020-01-01"]}),
            factor_returns=pd.DataFrame({"date": ["2020-01-01"], "equity": [0.01]}),
            window=3,
        )


def test_estimate_rolling_factor_exposures_rejects_no_factor_columns() -> None:
    with pytest.raises(RollingFactorExposureError, match="At least one factor"):
        estimate_rolling_factor_exposures(
            strategy_returns=pd.DataFrame(
                {
                    "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                    "return": [0.01, 0.02, 0.03],
                }
            ),
            factor_returns=pd.DataFrame(
                {"date": ["2020-01-01", "2020-01-02", "2020-01-03"]}
            ),
            window=3,
        )


def test_latest_exposures_rejects_empty_result() -> None:
    result = estimate_rolling_factor_exposures(
        strategy_returns=pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "return": [0.01, None, None],
            }
        ),
        factor_returns=pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "equity": [0.01, None, None],
            }
        ),
        window=3,
        min_observations=3,
    )

    with pytest.raises(RollingFactorExposureError, match="No rolling exposures"):
        result.latest_exposures()
