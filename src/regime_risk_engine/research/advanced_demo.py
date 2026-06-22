from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from regime_risk_engine.research.crisis_windows import (
    filter_crisis_windows_for_index,
)


class AdvancedResearchDemoInputError(ValueError):
    """Raised when advanced research demo inputs cannot be created."""


@dataclass(frozen=True, slots=True)
class AdvancedResearchDemoInputResult:
    """Generated advanced research demo input paths."""

    output_dir: Path
    price_data_path: Path
    static_weights_path: Path
    regime_policy_path: Path
    stress_periods_path: Path
    factor_returns_path: Path


def create_advanced_research_demo_inputs(
    output_dir: str | Path,
    overwrite: bool = True,
) -> AdvancedResearchDemoInputResult:
    """Create demo CSV inputs for the advanced research export workflow."""
    clean_output_dir = _prepare_output_dir(output_dir)

    result = AdvancedResearchDemoInputResult(
        output_dir=clean_output_dir,
        price_data_path=clean_output_dir / "prices.csv",
        static_weights_path=clean_output_dir / "static_weights.csv",
        regime_policy_path=clean_output_dir / "regime_policy.csv",
        stress_periods_path=clean_output_dir / "stress_periods.csv",
        factor_returns_path=clean_output_dir / "factor_returns.csv",
    )

    _validate_overwrite_policy(
        paths=[
            result.price_data_path,
            result.static_weights_path,
            result.regime_policy_path,
            result.stress_periods_path,
            result.factor_returns_path,
        ],
        overwrite=overwrite,
    )

    price_data = _build_demo_price_data()
    static_weights = _build_static_weights()
    regime_policy = _build_regime_policy()
    price_date_index = _extract_price_date_index(price_data)
    stress_periods = _build_demo_stress_periods(price_date_index)
    factor_returns = _build_factor_returns()

    price_data.to_csv(result.price_data_path, index=False)
    static_weights.to_csv(result.static_weights_path, index=False)
    regime_policy.to_csv(result.regime_policy_path, index=False)
    stress_periods.to_csv(result.stress_periods_path, index=False)
    factor_returns.to_csv(result.factor_returns_path, index=False)

    return result


def create_advanced_demo_inputs(
    output_dir: str | Path,
    overwrite: bool = True,
) -> AdvancedResearchDemoInputResult:
    """Create demo CSV inputs using the shorter public helper name."""
    return create_advanced_research_demo_inputs(
        output_dir=output_dir,
        overwrite=overwrite,
    )


def format_advanced_demo_input_result(
    result: AdvancedResearchDemoInputResult,
) -> str:
    """Format generated demo input paths for CLI-style output."""
    return "\n".join(
        [
            "Advanced research demo inputs created successfully.",
            f"Output directory: {result.output_dir}",
            f"Price data: {result.price_data_path}",
            f"Static weights: {result.static_weights_path}",
            f"Regime policy: {result.regime_policy_path}",
            f"Stress periods: {result.stress_periods_path}",
            f"Factor returns: {result.factor_returns_path}",
        ]
    )


def _prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser().resolve()

    if path.exists() and not path.is_dir():
        raise AdvancedResearchDemoInputError(
            f"Output path exists and is not a directory: {path}"
        )

    path.mkdir(parents=True, exist_ok=True)

    return path


def _validate_overwrite_policy(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return

    existing_paths = [path for path in paths if path.exists()]

    if existing_paths:
        existing = ", ".join(path.name for path in existing_paths)
        raise AdvancedResearchDemoInputError(
            f"Demo input file(s) already exist and overwrite=False: {existing}"
        )


def _build_demo_price_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=180, freq="D")
    prices = {
        "SPY": 100.0,
        "TLT": 100.0,
        "GLD": 100.0,
    }
    rows: list[dict[str, object]] = []

    for index, date in enumerate(dates):
        returns = _demo_asset_returns(index)

        for ticker, ticker_return in returns.items():
            prices[ticker] *= 1.0 + ticker_return
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "ticker": ticker,
                    "adjusted_close": prices[ticker],
                }
            )

    return pd.DataFrame(rows)


def _demo_asset_returns(index: int) -> dict[str, float]:
    if index < 60:
        return {
            "SPY": 0.0035,
            "TLT": 0.0004,
            "GLD": 0.0002,
        }

    if index < 120:
        return {
            "SPY": -0.0045,
            "TLT": 0.0028,
            "GLD": 0.0010,
        }

    return {
        "SPY": 0.0004,
        "TLT": -0.0012,
        "GLD": 0.0036,
    }


def _build_static_weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["SPY", "TLT", "GLD"],
            "weight": [0.60, 0.30, 0.10],
        }
    )


def _build_regime_policy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"regime": 0, "ticker": "SPY", "weight": 0.70},
            {"regime": 0, "ticker": "TLT", "weight": 0.20},
            {"regime": 0, "ticker": "GLD", "weight": 0.10},
            {"regime": 1, "ticker": "SPY", "weight": 0.25},
            {"regime": 1, "ticker": "TLT", "weight": 0.55},
            {"regime": 1, "ticker": "GLD", "weight": 0.20},
            {"regime": 2, "ticker": "SPY", "weight": 0.35},
            {"regime": 2, "ticker": "TLT", "weight": 0.20},
            {"regime": 2, "ticker": "GLD", "weight": 0.45},
        ]
    )


def _extract_price_date_index(price_data: pd.DataFrame) -> pd.DatetimeIndex:
    if "date" not in price_data.columns:
        raise AdvancedResearchDemoInputError(
            "Demo price data must contain a 'date' column."
        )

    dates = pd.to_datetime(
        price_data["date"].drop_duplicates(),
        errors="raise",
    )

    if dates.empty:
        raise AdvancedResearchDemoInputError(
            "Demo price data must contain at least one date."
        )

    return pd.DatetimeIndex(dates)


def _build_demo_stress_periods(price_index: pd.Index) -> pd.DataFrame:
    """Build demo stress periods from historical crisis presets.

    The advanced export workflow expects columns:
    name, start_date, end_date.
    """
    crisis_windows = filter_crisis_windows_for_index(price_index)

    if crisis_windows:
        return pd.DataFrame(
            [
                {
                    "name": window.name,
                    "start_date": window.start_date,
                    "end_date": window.end_date,
                }
                for window in crisis_windows
            ],
            columns=["name", "start_date", "end_date"],
        )

    date_index = pd.DatetimeIndex(price_index)

    if date_index.empty:
        raise AdvancedResearchDemoInputError(
            "price_index must contain at least one date."
        )

    start = date_index[int(len(date_index) * 0.35)]
    end = date_index[int(len(date_index) * 0.50)]

    return pd.DataFrame(
        [
            {
                "name": "demo_stress_window",
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
            }
        ],
        columns=["name", "start_date", "end_date"],
    )


def _build_stress_periods() -> pd.DataFrame:
    """Build stress periods for backward-compatible internal callers."""
    price_data = _build_demo_price_data()
    price_date_index = _extract_price_date_index(price_data)

    return _build_demo_stress_periods(price_date_index)


def _build_factor_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=180, freq="D")
    rows: list[dict[str, object]] = []

    for index, date in enumerate(dates):
        if index < 60:
            equity = 0.0032
            defensive = 0.0003
            real_asset = 0.0002
        elif index < 120:
            equity = -0.0042
            defensive = 0.0025
            real_asset = 0.0010
        else:
            equity = 0.0003
            defensive = -0.0010
            real_asset = 0.0034

        rows.append(
            {
                "date": date.date().isoformat(),
                "equity": equity,
                "defensive": defensive,
                "real_asset": real_asset,
            }
        )

    return pd.DataFrame(rows)
