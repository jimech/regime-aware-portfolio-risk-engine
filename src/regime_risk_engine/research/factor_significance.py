from dataclasses import dataclass
from math import erfc

import numpy as np
import pandas as pd


class FactorSignificanceError(ValueError):
    """Raised when factor significance diagnostics cannot be computed."""


@dataclass(frozen=True, slots=True)
class FactorSignificanceResult:
    """Factor beta significance diagnostics."""

    significance_table: pd.DataFrame
    alpha: float
    r_squared: float
    observations: int


def estimate_factor_significance(
    strategy_returns: pd.Series | pd.DataFrame,
    factor_returns: pd.DataFrame,
    strategy_name: str = "strategy",
    alpha: float = 0.05,
    include_intercept: bool = True,
) -> FactorSignificanceResult:
    """Estimate factor beta significance using OLS diagnostics."""
    clean_strategy = _prepare_strategy_returns(strategy_returns)
    clean_factors = _prepare_factor_returns(factor_returns)

    common_index = clean_strategy.index.intersection(clean_factors.index)

    if common_index.empty:
        raise FactorSignificanceError(
            "Strategy returns and factor returns have no overlapping dates"
        )

    y = clean_strategy.loc[common_index].astype(float)
    x = clean_factors.loc[common_index].astype(float)

    frame = pd.concat([y.rename("strategy_return"), x], axis=1).dropna()

    if frame.empty:
        raise FactorSignificanceError("No complete observations available")

    y_values = frame["strategy_return"].to_numpy(dtype=float)
    factor_columns = list(x.columns)
    x_values = frame[factor_columns].to_numpy(dtype=float)

    if include_intercept:
        design = np.column_stack([np.ones(len(x_values)), x_values])
        coefficient_names = ["alpha", *factor_columns]
    else:
        design = x_values
        coefficient_names = factor_columns

    observations = len(y_values)
    coefficient_count = design.shape[1]

    if observations <= coefficient_count:
        raise FactorSignificanceError(
            "Observations must exceed the number of estimated coefficients"
        )

    coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)
    fitted = design @ coefficients
    residuals = y_values - fitted

    residual_degrees_of_freedom = observations - coefficient_count
    residual_variance = float((residuals @ residuals) / residual_degrees_of_freedom)

    covariance = residual_variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.diag(covariance))

    t_stats = np.divide(
        coefficients,
        standard_errors,
        out=np.zeros_like(coefficients),
        where=standard_errors != 0,
    )

    p_values = _normal_approximation_two_sided_p_values(t_stats)

    rows = []

    for name, beta, standard_error, t_stat, p_value in zip(
        coefficient_names,
        coefficients,
        standard_errors,
        t_stats,
        p_values,
        strict=True,
    ):
        if name == "alpha":
            continue

        rows.append(
            {
                "strategy": strategy_name,
                "factor": name,
                "beta": float(beta),
                "standard_error": float(standard_error),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < alpha),
            }
        )

    total_sum_of_squares = float(((y_values - y_values.mean()) ** 2).sum())
    residual_sum_of_squares = float((residuals**2).sum())

    if total_sum_of_squares == 0:
        r_squared = 0.0
    else:
        r_squared = 1.0 - residual_sum_of_squares / total_sum_of_squares

    intercept = float(coefficients[0]) if include_intercept else 0.0

    return FactorSignificanceResult(
        significance_table=pd.DataFrame(rows),
        alpha=intercept,
        r_squared=float(r_squared),
        observations=observations,
    )


def _prepare_strategy_returns(
    strategy_returns: pd.Series | pd.DataFrame,
) -> pd.Series:
    if isinstance(strategy_returns, pd.Series):
        series = strategy_returns.copy()
    else:
        if "return" not in strategy_returns.columns:
            raise FactorSignificanceError(
                "Strategy returns DataFrame must contain a return column"
            )

        series = strategy_returns["return"].copy()

        if "date" in strategy_returns.columns:
            series.index = pd.to_datetime(strategy_returns["date"])

    series = pd.to_numeric(series, errors="coerce")
    series.index = pd.to_datetime(series.index)

    return series.sort_index()


def _prepare_factor_returns(factor_returns: pd.DataFrame) -> pd.DataFrame:
    if factor_returns.empty:
        raise FactorSignificanceError("Factor returns cannot be empty")

    frame = factor_returns.copy()

    if "date" in frame.columns:
        frame.index = pd.to_datetime(frame["date"])
        frame = frame.drop(columns=["date"])

    if frame.empty:
        raise FactorSignificanceError(
            "Factor returns must include at least one factor column"
        )

    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame.index = pd.to_datetime(frame.index)

    return frame.sort_index()


def _normal_approximation_two_sided_p_values(t_stats: np.ndarray) -> np.ndarray:
    values = [erfc(float(abs(t_stat)) / np.sqrt(2.0)) for t_stat in t_stats]

    return np.asarray(values, dtype=float)
