from dataclasses import dataclass
from math import sqrt
from typing import TypedDict

import numpy as np
import pandas as pd


class FactorExposureError(ValueError):
    """Raised when factor exposure analysis cannot be completed."""


@dataclass(frozen=True, slots=True)
class FactorExposureSummary:
    """Factor exposure analysis result."""

    exposure_table: pd.DataFrame
    regime_exposure_table: pd.DataFrame | None
    dominant_factor_by_strategy: pd.DataFrame
    narrative: str


class FactorRegressionResult(TypedDict):
    """Typed factor regression outputs."""

    alpha: float
    annualized_alpha: float
    r_squared: float
    residual_volatility: float
    betas: dict[str, float]


def build_factor_exposure_summary(
    strategy_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    regime_labels: pd.Series | None = None,
    annualization_factor: int = 252,
) -> FactorExposureSummary:
    """Estimate strategy factor exposures using linear regression."""
    _validate_inputs(
        strategy_returns=strategy_returns,
        factor_returns=factor_returns,
        regime_labels=regime_labels,
        annualization_factor=annualization_factor,
    )

    aligned_strategies, aligned_factors = _align_strategy_and_factor_returns(
        strategy_returns=strategy_returns,
        factor_returns=factor_returns,
    )

    exposure_table = _build_exposure_table(
        strategy_returns=aligned_strategies,
        factor_returns=aligned_factors,
        annualization_factor=annualization_factor,
    )

    aligned_regimes = None

    if regime_labels is not None:
        aligned_regimes = _align_regime_labels(
            regime_labels=regime_labels,
            index=aligned_strategies.index,
        )

    regime_exposure_table = None

    if aligned_regimes is not None:
        regime_exposure_table = _build_regime_exposure_table(
            strategy_returns=aligned_strategies,
            factor_returns=aligned_factors,
            regime_labels=aligned_regimes,
            annualization_factor=annualization_factor,
        )

    dominant_factor_by_strategy = _build_dominant_factor_table(exposure_table)

    narrative = _build_narrative(
        exposure_table=exposure_table,
        regime_exposure_table=regime_exposure_table,
        dominant_factor_by_strategy=dominant_factor_by_strategy,
    )

    return FactorExposureSummary(
        exposure_table=exposure_table,
        regime_exposure_table=regime_exposure_table,
        dominant_factor_by_strategy=dominant_factor_by_strategy,
        narrative=narrative,
    )


def _validate_inputs(
    strategy_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    regime_labels: pd.Series | None,
    annualization_factor: int,
) -> None:
    if strategy_returns.empty:
        raise FactorExposureError("Strategy returns cannot be empty")

    if factor_returns.empty:
        raise FactorExposureError("Factor returns cannot be empty")

    if not isinstance(strategy_returns.index, pd.DatetimeIndex):
        raise FactorExposureError("Strategy returns index must be a DatetimeIndex")

    if not isinstance(factor_returns.index, pd.DatetimeIndex):
        raise FactorExposureError("Factor returns index must be a DatetimeIndex")

    if regime_labels is not None and not isinstance(
        regime_labels.index,
        pd.DatetimeIndex,
    ):
        raise FactorExposureError("Regime labels index must be a DatetimeIndex")

    if annualization_factor <= 0:
        raise FactorExposureError("Annualization factor must be positive")

    _validate_numeric_frame(strategy_returns, "Strategy returns")
    _validate_numeric_frame(factor_returns, "Factor returns")

    if regime_labels is not None and regime_labels.isna().any():
        raise FactorExposureError("Regime labels cannot contain missing values")


def _validate_numeric_frame(frame: pd.DataFrame, label: str) -> None:
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")

    if numeric_frame.isna().any().any():
        raise FactorExposureError(f"{label} must be numeric and complete")


def _align_strategy_and_factor_returns(
    strategy_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_index = strategy_returns.index.intersection(factor_returns.index)

    if common_index.empty:
        raise FactorExposureError(
            "Strategy and factor returns have no overlapping dates"
        )

    aligned_strategies = strategy_returns.loc[common_index].astype(float).copy()
    aligned_factors = factor_returns.loc[common_index].astype(float).copy()

    if len(aligned_strategies) <= len(aligned_factors.columns) + 1:
        raise FactorExposureError(
            "Not enough observations to estimate factor exposures"
        )

    return aligned_strategies, aligned_factors


def _align_regime_labels(
    regime_labels: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    common_index = index.intersection(regime_labels.index)

    if common_index.empty:
        raise FactorExposureError(
            "Regime labels have no overlapping dates with returns"
        )

    return regime_labels.loc[index].astype(int).copy()


def _build_exposure_table(
    strategy_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    annualization_factor: int,
) -> pd.DataFrame:
    rows = []

    for strategy in strategy_returns.columns:
        regression = _fit_factor_regression(
            y=strategy_returns[strategy],
            x=factor_returns,
            annualization_factor=annualization_factor,
        )

        rows.append(
            {
                "strategy": str(strategy),
                "alpha": regression["alpha"],
                "annualized_alpha": regression["annualized_alpha"],
                "r_squared": regression["r_squared"],
                "residual_volatility": regression["residual_volatility"],
                **{
                    f"{factor}_beta": regression["betas"][factor]
                    for factor in factor_returns.columns
                },
            }
        )

    return pd.DataFrame(rows)


def _build_regime_exposure_table(
    strategy_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    regime_labels: pd.Series,
    annualization_factor: int,
) -> pd.DataFrame:
    rows = []

    combined = pd.concat(
        [
            strategy_returns,
            factor_returns,
            regime_labels.rename("regime"),
        ],
        axis=1,
    )

    strategy_columns = [str(column) for column in strategy_returns.columns]
    factor_columns = [str(column) for column in factor_returns.columns]

    for regime, regime_frame in combined.groupby("regime"):
        if len(regime_frame) <= len(factor_columns) + 1:
            continue

        for strategy in strategy_columns:
            regression = _fit_factor_regression(
                y=regime_frame[strategy],
                x=regime_frame[factor_columns],
                annualization_factor=annualization_factor,
            )

            rows.append(
                {
                    "regime": int(regime),
                    "strategy": strategy,
                    "observation_count": int(len(regime_frame)),
                    "alpha": regression["alpha"],
                    "annualized_alpha": regression["annualized_alpha"],
                    "r_squared": regression["r_squared"],
                    "residual_volatility": regression["residual_volatility"],
                    **{
                        f"{factor}_beta": regression["betas"][factor]
                        for factor in factor_columns
                    },
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "regime",
                "strategy",
                "observation_count",
                "alpha",
                "annualized_alpha",
                "r_squared",
                "residual_volatility",
                *[f"{factor}_beta" for factor in factor_returns.columns],
            ]
        )

    return pd.DataFrame(rows).sort_values(["regime", "strategy"]).reset_index(drop=True)


def _fit_factor_regression(
    y: pd.Series,
    x: pd.DataFrame,
    annualization_factor: int,
) -> FactorRegressionResult:
    y_values = y.to_numpy(dtype=float)
    x_values = x.to_numpy(dtype=float)

    design = np.column_stack(
        [
            np.ones(len(x_values)),
            x_values,
        ]
    )

    coefficients, *_ = np.linalg.lstsq(design, y_values, rcond=None)

    fitted_values = design @ coefficients
    residuals = y_values - fitted_values

    total_sum_of_squares = float(np.sum((y_values - np.mean(y_values)) ** 2))
    residual_sum_of_squares = float(np.sum(residuals**2))

    if total_sum_of_squares == 0.0:
        r_squared = 0.0
    else:
        r_squared = 1.0 - residual_sum_of_squares / total_sum_of_squares

    if len(residuals) < 2:
        residual_volatility = 0.0
    else:
        residual_volatility = float(
            np.std(residuals, ddof=1) * sqrt(annualization_factor)
        )

    alpha = float(coefficients[0])
    annualized_alpha = float(alpha * annualization_factor)

    betas: dict[str, float] = {
        str(factor): float(beta)
        for factor, beta in zip(x.columns, coefficients[1:], strict=True)
    }

    return {
        "alpha": alpha,
        "annualized_alpha": annualized_alpha,
        "r_squared": float(r_squared),
        "residual_volatility": residual_volatility,
        "betas": betas,
    }


def _build_dominant_factor_table(exposure_table: pd.DataFrame) -> pd.DataFrame:
    beta_columns = [
        str(column)
        for column in exposure_table.columns
        if str(column).endswith("_beta")
    ]

    rows = []

    for _, row in exposure_table.iterrows():
        beta_values = {
            column.replace("_beta", ""): _to_float(row[column])
            for column in beta_columns
        }

        if not beta_values:
            dominant_factor = None
            dominant_beta = 0.0
        else:
            dominant_factor = max(
                beta_values,
                key=lambda factor: abs(beta_values[factor]),
            )
            dominant_beta = beta_values[dominant_factor]

        rows.append(
            {
                "strategy": str(row["strategy"]),
                "dominant_factor": dominant_factor,
                "dominant_beta": dominant_beta,
            }
        )

    return pd.DataFrame(rows)


def _build_narrative(
    exposure_table: pd.DataFrame,
    regime_exposure_table: pd.DataFrame | None,
    dominant_factor_by_strategy: pd.DataFrame,
) -> str:
    strategy_count = int(exposure_table["strategy"].nunique())
    beta_columns = [
        str(column)
        for column in exposure_table.columns
        if str(column).endswith("_beta")
    ]
    factor_count = len(beta_columns)

    narrative = (
        f"Factor exposure analysis estimated {factor_count} factor beta(s) "
        f"for {strategy_count} strategy return series."
    )

    if not dominant_factor_by_strategy.empty:
        dominant_parts = [
            f"{row['strategy']} was most exposed to {row['dominant_factor']}"
            for row in dominant_factor_by_strategy.to_dict(orient="records")
        ]
        narrative += " " + "; ".join(dominant_parts) + "."

    if regime_exposure_table is not None and not regime_exposure_table.empty:
        regime_count = int(regime_exposure_table["regime"].nunique())
        narrative += (
            f" Regime-conditioned factor exposures were estimated across "
            f"{regime_count} regimes."
        )

    return narrative


def _to_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise FactorExposureError("Expected numeric factor exposure value")

    return float(numeric)
