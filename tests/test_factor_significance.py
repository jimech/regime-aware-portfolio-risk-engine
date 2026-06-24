import pandas as pd
import pytest

from regime_risk_engine.research.factor_significance import (
    FactorSignificanceError,
    estimate_factor_significance,
)


def test_estimate_factor_significance_identifies_meaningful_beta() -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    equity = pd.Series(
        [0.01, 0.02, -0.01, 0.03, -0.02] * 8,
        index=dates,
    )
    strategy = 0.001 + 2.0 * equity

    result = estimate_factor_significance(
        strategy_returns=strategy,
        factor_returns=pd.DataFrame({"equity": equity}),
        strategy_name="dynamic",
    )

    table = result.significance_table

    assert result.observations == 40
    assert result.r_squared > 0.99
    assert table.loc[0, "strategy"] == "dynamic"
    assert table.loc[0, "factor"] == "equity"
    assert table.loc[0, "beta"] == pytest.approx(2.0)
    assert table.loc[0, "p_value"] < 0.05
    assert table.loc[0, "significant"]


def test_estimate_factor_significance_supports_dataframe_strategy_input() -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    equity = [0.01, 0.02, -0.01, 0.03, -0.02] * 6
    strategy = [0.001 + 1.5 * value for value in equity]

    result = estimate_factor_significance(
        strategy_returns=pd.DataFrame({"date": dates, "return": strategy}),
        factor_returns=pd.DataFrame({"date": dates, "equity": equity}),
    )

    table = result.significance_table

    assert table.loc[0, "factor"] == "equity"
    assert table.loc[0, "beta"] == pytest.approx(1.5)


def test_estimate_factor_significance_rejects_missing_return_column() -> None:
    with pytest.raises(FactorSignificanceError, match="return column"):
        estimate_factor_significance(
            strategy_returns=pd.DataFrame({"not_return": [0.01, 0.02]}),
            factor_returns=pd.DataFrame({"equity": [0.01, 0.02]}),
        )


def test_estimate_factor_significance_rejects_empty_factors() -> None:
    with pytest.raises(FactorSignificanceError, match="cannot be empty"):
        estimate_factor_significance(
            strategy_returns=pd.Series([0.01, 0.02]),
            factor_returns=pd.DataFrame(),
        )


def test_estimate_factor_significance_rejects_non_overlapping_dates() -> None:
    strategy = pd.Series(
        [0.01, 0.02, 0.03],
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    factors = pd.DataFrame(
        {"equity": [0.01, 0.02, 0.03]},
        index=pd.date_range("2021-01-01", periods=3, freq="D"),
    )

    with pytest.raises(FactorSignificanceError, match="no overlapping dates"):
        estimate_factor_significance(
            strategy_returns=strategy,
            factor_returns=factors,
        )


def test_estimate_factor_significance_rejects_too_few_observations() -> None:
    with pytest.raises(FactorSignificanceError, match="Observations must exceed"):
        estimate_factor_significance(
            strategy_returns=pd.Series([0.01, 0.02]),
            factor_returns=pd.DataFrame({"equity": [0.01, 0.02]}),
        )
