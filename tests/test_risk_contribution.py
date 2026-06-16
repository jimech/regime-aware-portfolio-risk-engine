import pandas as pd
import pytest

from regime_risk_engine.risk.contribution import (
    RiskContributionError,
    calculate_component_risk_contribution,
    calculate_marginal_risk_contribution,
    calculate_percentage_risk_contribution,
    calculate_portfolio_volatility,
    summarize_regime_risk_contributions,
    summarize_risk_contributions,
)


def make_covariance_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY": [0.04, 0.01],
            "TLT": [0.01, 0.09],
        },
        index=["SPY", "TLT"],
    )


def make_weights() -> dict[str, float]:
    return {
        "SPY": 0.6,
        "TLT": 0.4,
    }


def test_calculate_portfolio_volatility() -> None:
    covariance_matrix = make_covariance_matrix()
    weights = make_weights()

    volatility = calculate_portfolio_volatility(covariance_matrix, weights)

    expected_variance = (0.6**2 * 0.04) + (0.4**2 * 0.09) + (2 * 0.6 * 0.4 * 0.01)

    assert volatility == pytest.approx(expected_variance**0.5)


def test_calculate_marginal_risk_contribution() -> None:
    covariance_matrix = make_covariance_matrix()
    weights = make_weights()

    marginal = calculate_marginal_risk_contribution(covariance_matrix, weights)

    assert list(marginal.index) == ["SPY", "TLT"]
    assert marginal.name == "marginal_risk_contribution"


def test_calculate_component_risk_contribution_sums_to_volatility() -> None:
    covariance_matrix = make_covariance_matrix()
    weights = make_weights()

    component = calculate_component_risk_contribution(covariance_matrix, weights)
    volatility = calculate_portfolio_volatility(covariance_matrix, weights)

    assert component.sum() == pytest.approx(volatility)


def test_calculate_percentage_risk_contribution_sums_to_one() -> None:
    covariance_matrix = make_covariance_matrix()
    weights = make_weights()

    percentage = calculate_percentage_risk_contribution(covariance_matrix, weights)

    assert percentage.sum() == pytest.approx(1.0)


def test_summarize_risk_contributions() -> None:
    covariance_matrix = make_covariance_matrix()
    weights = make_weights()

    summary = summarize_risk_contributions(covariance_matrix, weights)

    assert list(summary.columns) == [
        "ticker",
        "weight",
        "marginal_risk_contribution",
        "component_risk_contribution",
        "percentage_risk_contribution",
    ]
    assert summary["ticker"].tolist() == ["SPY", "TLT"]
    assert summary["component_risk_contribution"].sum() == pytest.approx(
        calculate_portfolio_volatility(covariance_matrix, weights)
    )
    assert summary["percentage_risk_contribution"].sum() == pytest.approx(1.0)


def test_summarize_regime_risk_contributions() -> None:
    covariance_matrices = {
        0: make_covariance_matrix(),
        1: make_covariance_matrix() * 2.0,
    }
    weights = make_weights()

    summary = summarize_regime_risk_contributions(covariance_matrices, weights)

    assert set(summary["regime"]) == {0, 1}
    assert set(summary["ticker"]) == {"SPY", "TLT"}
    assert len(summary) == 4


def test_risk_contribution_rejects_empty_covariance_matrix() -> None:
    covariance_matrix = pd.DataFrame()
    weights = make_weights()

    with pytest.raises(RiskContributionError, match="empty"):
        calculate_portfolio_volatility(covariance_matrix, weights)


def test_risk_contribution_rejects_non_square_covariance_matrix() -> None:
    covariance_matrix = pd.DataFrame(
        {
            "SPY": [0.04, 0.01],
            "TLT": [0.01, 0.09],
            "GLD": [0.01, 0.02],
        },
        index=["SPY", "TLT"],
    )
    weights = {
        "SPY": 0.5,
        "TLT": 0.3,
        "GLD": 0.2,
    }

    with pytest.raises(RiskContributionError, match="square"):
        calculate_portfolio_volatility(covariance_matrix, weights)


def test_risk_contribution_rejects_non_symmetric_covariance_matrix() -> None:
    covariance_matrix = pd.DataFrame(
        {
            "SPY": [0.04, 0.02],
            "TLT": [0.01, 0.09],
        },
        index=["SPY", "TLT"],
    )
    weights = make_weights()

    with pytest.raises(RiskContributionError, match="symmetric"):
        calculate_portfolio_volatility(covariance_matrix, weights)


def test_risk_contribution_rejects_missing_weights() -> None:
    covariance_matrix = make_covariance_matrix()

    with pytest.raises(RiskContributionError, match="Missing weight"):
        calculate_portfolio_volatility(
            covariance_matrix,
            weights={"SPY": 1.0},
        )


def test_risk_contribution_rejects_unknown_weight_ticker() -> None:
    covariance_matrix = make_covariance_matrix()

    with pytest.raises(RiskContributionError, match="unknown ticker"):
        calculate_portfolio_volatility(
            covariance_matrix,
            weights={
                "SPY": 0.5,
                "TLT": 0.4,
                "GLD": 0.1,
            },
        )


def test_risk_contribution_rejects_negative_weights() -> None:
    covariance_matrix = make_covariance_matrix()

    with pytest.raises(RiskContributionError, match="non-negative"):
        calculate_portfolio_volatility(
            covariance_matrix,
            weights={
                "SPY": 1.1,
                "TLT": -0.1,
            },
        )


def test_risk_contribution_rejects_weights_that_do_not_sum_to_one() -> None:
    covariance_matrix = make_covariance_matrix()

    with pytest.raises(RiskContributionError, match="sum to 1.0"):
        calculate_portfolio_volatility(
            covariance_matrix,
            weights={
                "SPY": 0.7,
                "TLT": 0.7,
            },
        )


def test_summarize_regime_risk_contributions_rejects_empty_mapping() -> None:
    with pytest.raises(RiskContributionError, match="At least one covariance"):
        summarize_regime_risk_contributions({}, make_weights())
