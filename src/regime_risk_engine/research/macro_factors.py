from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

MacroTransformation = Literal[
    "level",
    "difference",
    "percent_change",
    "log_difference",
]
MacroFillMethod = Literal["ffill", "bfill", "none"]


class MacroFactorInputError(ValueError):
    """Raised when macro factor inputs cannot be transformed."""


@dataclass(frozen=True, slots=True)
class MacroFactorSpec:
    """Configuration for transforming one macroeconomic input series."""

    name: str
    source_column: str
    transformation: MacroTransformation = "level"
    lag_periods: int = 0
    standardize: bool = False
    fill_method: MacroFillMethod = "ffill"

    def to_record(self) -> dict[str, object]:
        """Return a serializable representation of the factor spec."""
        return {
            "name": self.name,
            "source_column": self.source_column,
            "transformation": self.transformation,
            "lag_periods": self.lag_periods,
            "standardize": self.standardize,
            "fill_method": self.fill_method,
        }


def build_macro_factor_matrix(
    data: pd.DataFrame,
    specs: list[MacroFactorSpec],
    date_column: str = "date",
) -> pd.DataFrame:
    """Transform raw macroeconomic time series into model-ready factors."""
    _require_columns(data, {date_column}, "Macro factor input")

    if not specs:
        raise MacroFactorInputError("At least one macro factor spec is required.")

    dates = pd.to_datetime(data[date_column], errors="raise")
    result = pd.DataFrame({"date": dates})

    for spec in specs:
        _validate_spec(spec)
        _require_columns(data, {spec.source_column}, "Macro factor input")

        raw_series = pd.to_numeric(data[spec.source_column], errors="coerce")
        filled_series = _apply_fill_method(raw_series, spec.fill_method)
        transformed_series = _transform_macro_series(
            filled_series,
            transformation=spec.transformation,
            source_column=spec.source_column,
        )

        if spec.lag_periods > 0:
            transformed_series = transformed_series.shift(spec.lag_periods)

        if spec.standardize:
            transformed_series = _standardize_series(
                transformed_series,
                factor_name=spec.name,
            )

        result[spec.name] = transformed_series.astype(float)

    return result.sort_values("date").reset_index(drop=True)


def infer_macro_factor_specs(
    data: pd.DataFrame,
    date_column: str = "date",
) -> list[MacroFactorSpec]:
    """Infer level-factor specs from numeric columns in a macro dataset."""
    _require_columns(data, {date_column}, "Macro factor input")

    specs: list[MacroFactorSpec] = []

    for column in data.columns:
        if column == date_column:
            continue

        numeric_series = pd.to_numeric(data[column], errors="coerce")

        if numeric_series.notna().any():
            specs.append(
                MacroFactorSpec(
                    name=str(column),
                    source_column=str(column),
                    transformation="level",
                )
            )

    if not specs:
        raise MacroFactorInputError("No numeric macro factor columns were found.")

    return specs


def read_macro_factor_csv(
    input_path: str | Path,
    specs: list[MacroFactorSpec],
    date_column: str = "date",
) -> pd.DataFrame:
    """Read a macro CSV and transform it into a factor matrix."""
    frame = pd.read_csv(input_path)

    return build_macro_factor_matrix(
        data=frame,
        specs=specs,
        date_column=date_column,
    )


def write_macro_factor_matrix_csv(
    output_path: str | Path,
    data: pd.DataFrame,
    specs: list[MacroFactorSpec],
    date_column: str = "date",
) -> Path:
    """Transform macro factors and write the factor matrix to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    factor_matrix = build_macro_factor_matrix(
        data=data,
        specs=specs,
        date_column=date_column,
    )
    factor_matrix.to_csv(path, index=False)

    return path


def _validate_spec(spec: MacroFactorSpec) -> None:
    if not spec.name.strip():
        raise MacroFactorInputError("Macro factor spec name cannot be empty.")

    if not spec.source_column.strip():
        raise MacroFactorInputError("Macro factor source_column cannot be empty.")

    if spec.lag_periods < 0:
        raise MacroFactorInputError("lag_periods must be non-negative.")

    if spec.transformation not in {
        "level",
        "difference",
        "percent_change",
        "log_difference",
    }:
        raise MacroFactorInputError(
            f"Unsupported macro factor transformation: {spec.transformation}"
        )

    if spec.fill_method not in {"ffill", "bfill", "none"}:
        raise MacroFactorInputError(
            f"Unsupported macro factor fill method: {spec.fill_method}"
        )


def _require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    label: str,
) -> None:
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise MacroFactorInputError(f"{label} missing column(s): {missing}")


def _apply_fill_method(
    series: pd.Series,
    fill_method: MacroFillMethod,
) -> pd.Series:
    if fill_method == "ffill":
        return series.ffill()

    if fill_method == "bfill":
        return series.bfill()

    if fill_method == "none":
        return series

    raise MacroFactorInputError(f"Unsupported macro factor fill method: {fill_method}")


def _transform_macro_series(
    series: pd.Series,
    transformation: MacroTransformation,
    source_column: str,
) -> pd.Series:
    if transformation == "level":
        return series

    if transformation == "difference":
        return series.diff()

    if transformation == "percent_change":
        return series.pct_change(fill_method=None)

    if transformation == "log_difference":
        non_missing = series.dropna()

        if (non_missing <= 0).any():
            raise MacroFactorInputError(
                "log_difference transformation requires positive values "
                f"for column '{source_column}'."
            )

        return pd.Series(
            np.log(series.to_numpy(dtype=float)),
            index=series.index,
        ).diff()

    raise MacroFactorInputError(
        f"Unsupported macro factor transformation: {transformation}"
    )


def _standardize_series(
    series: pd.Series,
    factor_name: str,
) -> pd.Series:
    mean = float(series.mean(skipna=True))
    std = float(series.std(skipna=True))

    if np.isnan(std) or std == 0.0:
        raise MacroFactorInputError(
            f"Cannot standardize macro factor '{factor_name}' with zero "
            "or undefined standard deviation."
        )

    return (series - mean) / std
