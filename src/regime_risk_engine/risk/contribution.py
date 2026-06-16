from collections.abc import Mapping

import numpy as np
import pandas as pd


class RiskContributionError(ValueError):
    """Raised when risk contribution analytics cannot be calculated."""


RISK_CONTRIBUTION_COLUMNS = [
    "ticker",
    "weight",
    "marginal_risk_contribution",
    "component_risk_contribution",
    "percentage_risk_contribution",
]


def calculate_portfolio_volatility(
    covariance_matrix: pd.DataFrame,
    weights: Mapping[str, float],
) -> float:
    """Calculate portfolio volatility from a covariance matrix and weights."""
    aligned_covariance, weight_vector = _prepare_covariance_and_weights(
        covariance_matrix,
        weights,
    )

    variance = float(weight_vector.T @ aligned_covariance.to_numpy() @ weight_vector)

    if variance < 0:
        raise RiskContributionError("Portfolio variance cannot be negative")

    return float(np.sqrt(variance))


def calculate_marginal_risk_contribution(
    covariance_matrix: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.Series:
    """Calculate marginal contribution of each asset to portfolio volatility."""
    aligned_covariance, weight_vector = _prepare_covariance_and_weights(
        covariance_matrix,
        weights,
    )
    portfolio_volatility = calculate_portfolio_volatility(
        aligned_covariance,
        weights,
    )

    if portfolio_volatility == 0:
        raise RiskContributionError("Portfolio volatility must be greater than zero")

    marginal_contributions = aligned_covariance.to_numpy() @ weight_vector
    marginal_contributions = marginal_contributions / portfolio_volatility

    return pd.Series(
        marginal_contributions,
        index=aligned_covariance.index,
        name="marginal_risk_contribution",
    )


def calculate_component_risk_contribution(
    covariance_matrix: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.Series:
    """Calculate component contribution of each asset to portfolio volatility."""
    marginal_contributions = calculate_marginal_risk_contribution(
        covariance_matrix,
        weights,
    )
    weight_series = _weights_to_series(weights, list(marginal_contributions.index))

    component_contributions = weight_series * marginal_contributions
    component_contributions.name = "component_risk_contribution"

    return component_contributions


def calculate_percentage_risk_contribution(
    covariance_matrix: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.Series:
    """Calculate percentage contribution of each asset to portfolio volatility."""
    component_contributions = calculate_component_risk_contribution(
        covariance_matrix,
        weights,
    )
    portfolio_volatility = calculate_portfolio_volatility(
        covariance_matrix,
        weights,
    )

    if portfolio_volatility == 0:
        raise RiskContributionError("Portfolio volatility must be greater than zero")

    percentage_contributions = component_contributions / portfolio_volatility
    percentage_contributions.name = "percentage_risk_contribution"

    return percentage_contributions


def summarize_risk_contributions(
    covariance_matrix: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Build a full risk contribution summary table."""
    aligned_covariance, _ = _prepare_covariance_and_weights(
        covariance_matrix,
        weights,
    )
    tickers = list(aligned_covariance.index)

    weight_series = _weights_to_series(weights, tickers)
    marginal = calculate_marginal_risk_contribution(
        aligned_covariance,
        weights,
    )
    component = calculate_component_risk_contribution(
        aligned_covariance,
        weights,
    )
    percentage = calculate_percentage_risk_contribution(
        aligned_covariance,
        weights,
    )

    summary = pd.DataFrame(
        {
            "ticker": tickers,
            "weight": weight_series.loc[tickers].to_numpy(),
            "marginal_risk_contribution": marginal.loc[tickers].to_numpy(),
            "component_risk_contribution": component.loc[tickers].to_numpy(),
            "percentage_risk_contribution": percentage.loc[tickers].to_numpy(),
        }
    )

    return summary[RISK_CONTRIBUTION_COLUMNS]


def summarize_regime_risk_contributions(
    covariance_matrices: Mapping[int, pd.DataFrame],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Calculate risk contribution summaries for each regime."""
    if not covariance_matrices:
        raise RiskContributionError("At least one covariance matrix is required")

    rows: list[pd.DataFrame] = []

    for regime, covariance_matrix in covariance_matrices.items():
        summary = summarize_risk_contributions(
            covariance_matrix=covariance_matrix,
            weights=weights,
        )
        summary.insert(0, "regime", int(regime))
        rows.append(summary)

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["regime", "ticker"])
        .reset_index(drop=True)
    )


def _prepare_covariance_and_weights(
    covariance_matrix: pd.DataFrame,
    weights: Mapping[str, float],
) -> tuple[pd.DataFrame, np.ndarray]:
    _validate_covariance_matrix(covariance_matrix)

    tickers = [str(ticker).upper().strip() for ticker in covariance_matrix.index]
    aligned_covariance = covariance_matrix.copy()
    aligned_covariance.index = tickers
    aligned_covariance.columns = [
        str(ticker).upper().strip() for ticker in covariance_matrix.columns
    ]

    if list(aligned_covariance.index) != list(aligned_covariance.columns):
        raise RiskContributionError(
            "Covariance matrix index and columns must contain the same tickers"
        )

    clean_weights = _validate_weights(weights, tickers)
    weight_vector = np.array([clean_weights[ticker] for ticker in tickers], dtype=float)

    return aligned_covariance, weight_vector


def _validate_covariance_matrix(covariance_matrix: pd.DataFrame) -> None:
    if covariance_matrix.empty:
        raise RiskContributionError("Covariance matrix is empty")

    if covariance_matrix.index.has_duplicates:
        raise RiskContributionError(
            "Covariance matrix index contains duplicate tickers"
        )

    if covariance_matrix.columns.has_duplicates:
        raise RiskContributionError(
            "Covariance matrix columns contain duplicate tickers"
        )

    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise RiskContributionError("Covariance matrix must be square")

    if covariance_matrix.isna().any().any():
        raise RiskContributionError("Covariance matrix contains missing values")

    values = covariance_matrix.to_numpy(dtype=float)

    if not np.allclose(values, values.T, atol=1e-10):
        raise RiskContributionError("Covariance matrix must be symmetric")


def _validate_weights(
    weights: Mapping[str, float],
    tickers: list[str],
) -> dict[str, float]:
    if not weights:
        raise RiskContributionError("Portfolio weights cannot be empty")

    clean_weights = {
        str(ticker).upper().strip(): float(weight) for ticker, weight in weights.items()
    }

    expected_tickers = set(tickers)
    weight_tickers = set(clean_weights)

    missing_tickers = sorted(expected_tickers.difference(weight_tickers))
    extra_tickers = sorted(weight_tickers.difference(expected_tickers))

    if missing_tickers:
        missing = ", ".join(missing_tickers)
        raise RiskContributionError(f"Missing weight(s) for ticker(s): {missing}")

    if extra_tickers:
        extra = ", ".join(extra_tickers)
        raise RiskContributionError(f"Weight provided for unknown ticker(s): {extra}")

    if any(weight < 0 for weight in clean_weights.values()):
        raise RiskContributionError("Portfolio weights must be non-negative")

    weight_sum = sum(clean_weights.values())

    if abs(weight_sum - 1.0) > 1e-8:
        raise RiskContributionError("Portfolio weights must sum to 1.0")

    return clean_weights


def _weights_to_series(
    weights: Mapping[str, float],
    tickers: list[str],
) -> pd.Series:
    clean_weights = _validate_weights(weights, tickers)

    return pd.Series(
        {ticker: clean_weights[ticker] for ticker in tickers},
        dtype=float,
        name="weight",
    )
