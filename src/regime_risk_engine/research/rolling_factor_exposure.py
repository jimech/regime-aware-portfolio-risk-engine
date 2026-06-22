from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class RollingFactorExposureError(ValueError):
    """Raised when rolling factor exposure analysis cannot be estimated."""


@dataclass(frozen=True, slots=True)
class RollingFactorExposureResult:
    """Rolling factor exposure analysis output."""

    exposures: pd.DataFrame
    factor_columns: tuple[str, ...]
    return_column: str
    window: int
    min_observations: int
    include_intercept: bool

    def latest_exposures(self) -> pd.Series:
        """Return the latest available rolling exposure row."""
        if self.exposures.empty:
            raise RollingFactorExposureError("No rolling exposures are available.")

        return self.exposures.iloc[-1]


def estimate_rolling_factor_exposures(
    strategy_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    return_column: str = "return",
    date_column: str = "date",
    factor_columns: list[str] | None = None,
    window: int = 60,
    min_observations: int | None = None,
    include_intercept: bool = True,
) -> RollingFactorExposureResult:
    """Estimate rolling factor betas using ordinary least squares."""
    if window <= 1:
        raise RollingFactorExposureError("window must be greater than 1.")

    clean_min_observations = window if min_observations is None else min_observations

    if clean_min_observations <= 1:
        raise RollingFactorExposureError("min_observations must be greater than 1.")

    _require_columns(strategy_returns, {date_column, return_column}, "Strategy returns")
    _require_columns(factor_returns, {date_column}, "Factor returns")

    selected_factor_columns = _resolve_factor_columns(
        factor_returns=factor_returns,
        date_column=date_column,
        factor_columns=factor_columns,
    )

    if clean_min_observations > window:
        raise RollingFactorExposureError(
            "min_observations must be less than or equal to window."
        )

    merged = _prepare_rolling_factor_data(
        strategy_returns=strategy_returns,
        factor_returns=factor_returns,
        return_column=return_column,
        date_column=date_column,
        factor_columns=selected_factor_columns,
    )

    minimum_required_observations = len(selected_factor_columns) + int(
        include_intercept
    )

    if clean_min_observations <= minimum_required_observations:
        raise RollingFactorExposureError(
            "min_observations must exceed the number of estimated coefficients."
        )

    exposure_rows: list[dict[str, object]] = []

    for end_position in range(window - 1, len(merged)):
        window_frame = merged.iloc[end_position - window + 1 : end_position + 1]
        window_frame = window_frame.dropna(
            subset=[return_column, *selected_factor_columns]
        )

        if len(window_frame) < clean_min_observations:
            continue

        row = _estimate_single_window_exposure(
            window_frame=window_frame,
            return_column=return_column,
            factor_columns=selected_factor_columns,
            date_value=merged.iloc[end_position][date_column],
            include_intercept=include_intercept,
        )
        exposure_rows.append(row)

    exposures = pd.DataFrame(exposure_rows)

    return RollingFactorExposureResult(
        exposures=exposures,
        factor_columns=tuple(selected_factor_columns),
        return_column=return_column,
        window=window,
        min_observations=clean_min_observations,
        include_intercept=include_intercept,
    )


def read_rolling_factor_exposure_inputs(
    strategy_returns_path: str | Path,
    factor_returns_path: str | Path,
    return_column: str = "return",
    date_column: str = "date",
    factor_columns: list[str] | None = None,
    window: int = 60,
    min_observations: int | None = None,
    include_intercept: bool = True,
) -> RollingFactorExposureResult:
    """Read rolling factor exposure inputs from CSV files and estimate exposures."""
    strategy_returns = pd.read_csv(strategy_returns_path)
    factor_returns = pd.read_csv(factor_returns_path)

    return estimate_rolling_factor_exposures(
        strategy_returns=strategy_returns,
        factor_returns=factor_returns,
        return_column=return_column,
        date_column=date_column,
        factor_columns=factor_columns,
        window=window,
        min_observations=min_observations,
        include_intercept=include_intercept,
    )


def write_rolling_factor_exposures_csv(
    output_path: str | Path,
    result: RollingFactorExposureResult,
) -> Path:
    """Write rolling factor exposures to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    result.exposures.to_csv(path, index=False)

    return path


def summarize_rolling_factor_exposures(
    result: RollingFactorExposureResult,
) -> pd.DataFrame:
    """Summarize rolling factor exposure ranges and latest values."""
    if result.exposures.empty:
        raise RollingFactorExposureError("No rolling exposures are available.")

    rows: list[dict[str, object]] = []

    for factor in result.factor_columns:
        beta_column = f"{factor}_beta"
        beta_series = result.exposures[beta_column]

        rows.append(
            {
                "factor": factor,
                "latest_beta": float(beta_series.iloc[-1]),
                "average_beta": float(beta_series.mean()),
                "minimum_beta": float(beta_series.min()),
                "maximum_beta": float(beta_series.max()),
                "beta_volatility": float(beta_series.std(ddof=1)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "factor",
            "latest_beta",
            "average_beta",
            "minimum_beta",
            "maximum_beta",
            "beta_volatility",
        ],
    )


def _prepare_rolling_factor_data(
    strategy_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    return_column: str,
    date_column: str,
    factor_columns: list[str],
) -> pd.DataFrame:
    strategy_frame = strategy_returns[[date_column, return_column]].copy()
    factor_frame = factor_returns[[date_column, *factor_columns]].copy()

    strategy_frame[date_column] = pd.to_datetime(
        strategy_frame[date_column],
        errors="raise",
    )
    factor_frame[date_column] = pd.to_datetime(
        factor_frame[date_column],
        errors="raise",
    )

    strategy_frame[return_column] = pd.to_numeric(
        strategy_frame[return_column],
        errors="coerce",
    )

    for factor in factor_columns:
        factor_frame[factor] = pd.to_numeric(factor_frame[factor], errors="coerce")

    merged = strategy_frame.merge(
        factor_frame,
        on=date_column,
        how="inner",
    )

    if merged.empty:
        raise RollingFactorExposureError(
            "No overlapping dates found between strategy and factor returns."
        )

    return merged.sort_values(date_column).reset_index(drop=True)


def _estimate_single_window_exposure(
    window_frame: pd.DataFrame,
    return_column: str,
    factor_columns: list[str],
    date_value: object,
    include_intercept: bool,
) -> dict[str, object]:
    y = window_frame[return_column].to_numpy(dtype=float)
    x = window_frame[factor_columns].to_numpy(dtype=float)

    if include_intercept:
        x = np.column_stack([np.ones(len(x)), x])

    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    predictions = x @ coefficients
    residuals = y - predictions

    total_sum_squares = float(np.sum((y - y.mean()) ** 2.0))
    residual_sum_squares = float(np.sum(residuals**2.0))

    if total_sum_squares == 0.0:
        r_squared = 0.0
    else:
        r_squared = 1.0 - residual_sum_squares / total_sum_squares

    if include_intercept:
        alpha = float(coefficients[0])
        betas = coefficients[1:]
    else:
        alpha = 0.0
        betas = coefficients

    beta_values = {
        f"{factor}_beta": float(beta)
        for factor, beta in zip(factor_columns, betas, strict=True)
    }

    dominant_factor = max(
        beta_values,
        key=lambda column: abs(beta_values[column]),
    ).removesuffix("_beta")

    return {
        "date": pd.Timestamp(date_value).date().isoformat(),
        "alpha": alpha,
        **beta_values,
        "r_squared": float(r_squared),
        "observations": int(len(window_frame)),
        "residual_volatility": float(np.std(residuals, ddof=1)),
        "dominant_factor": dominant_factor,
    }


def _resolve_factor_columns(
    factor_returns: pd.DataFrame,
    date_column: str,
    factor_columns: list[str] | None,
) -> list[str]:
    if factor_columns is not None:
        if not factor_columns:
            raise RollingFactorExposureError("At least one factor column is required.")

        _require_columns(factor_returns, set(factor_columns), "Factor returns")
        return list(factor_columns)

    inferred_factor_columns = [
        column for column in factor_returns.columns if column != date_column
    ]

    if not inferred_factor_columns:
        raise RollingFactorExposureError("At least one factor column is required.")

    return inferred_factor_columns


def _require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    label: str,
) -> None:
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RollingFactorExposureError(f"{label} missing column(s): {missing}")
