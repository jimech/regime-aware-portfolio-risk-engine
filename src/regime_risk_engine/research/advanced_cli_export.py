from pathlib import Path

import pandas as pd

from regime_risk_engine.research.advanced_export import (
    AdvancedResearchExportResult,
    export_advanced_research_package,
)
from regime_risk_engine.research.advanced_memo import AdvancedResearchMemoConfig
from regime_risk_engine.research.advanced_workflow import (
    build_advanced_research_workflow,
)
from regime_risk_engine.research.market_workflow import (
    MarketResearchWorkflowConfig,
    run_market_research_workflow,
)
from regime_risk_engine.research.memo import MarketResearchMemoConfig
from regime_risk_engine.research.scenario_simulation import (
    RegimeScenarioSimulationConfig,
)
from regime_risk_engine.research.stress_testing import StressPeriod


class AdvancedResearchCliExportError(ValueError):
    """Raised when CLI advanced research export cannot be completed."""


def export_advanced_research_from_files(
    price_data_path: str | Path,
    static_weights_path: str | Path,
    output_dir: str | Path,
    regime_policy_path: str | Path | None = None,
    stress_periods_path: str | Path | None = None,
    factor_returns_path: str | Path | None = None,
    n_regimes: int = 3,
    feature_window: int = 21,
    transaction_cost_bps: float = 5.0,
    random_state: int = 42,
    scenario_horizon: int = 21,
    scenario_simulations: int = 1_000,
    rolling_factor_window: int = 20,
    analyst: str | None = None,
    overwrite: bool = True,
) -> AdvancedResearchExportResult:
    """Run the advanced research workflow from CSV files and export a package."""
    _validate_positive_int(n_regimes, "Number of regimes")
    _validate_positive_int(feature_window, "Feature window")
    _validate_positive_int(scenario_horizon, "Scenario horizon")
    _validate_positive_int(scenario_simulations, "Scenario simulations")
    _validate_positive_int(rolling_factor_window, "Rolling factor window")

    price_data = _read_price_data(price_data_path)
    static_weights = _read_static_weights(static_weights_path)

    if regime_policy_path is None:
        regime_policy = _build_static_regime_policy(
            static_weights=static_weights,
            n_regimes=n_regimes,
        )
    else:
        regime_policy = _read_regime_policy(
            path=regime_policy_path,
            static_weights=static_weights,
        )

    stress_periods = None

    if stress_periods_path is not None:
        stress_periods = _read_stress_periods(stress_periods_path)

    factor_returns = None

    if factor_returns_path is not None:
        factor_returns = _read_factor_returns(factor_returns_path)

    market_result = run_market_research_workflow(
        price_data=price_data,
        static_weights=static_weights,
        regime_weight_policy=regime_policy,
        config=MarketResearchWorkflowConfig(
            n_regimes=n_regimes,
            feature_window=feature_window,
            transaction_cost_bps=transaction_cost_bps,
            random_state=random_state,
        ),
    )

    market_memo_config = MarketResearchMemoConfig(
        analyst=analyst,
    )
    advanced_memo_config = AdvancedResearchMemoConfig(
        analyst=analyst,
    )

    advanced_result = build_advanced_research_workflow(
        market_result=market_result,
        stress_periods=stress_periods,
        factor_returns=factor_returns,
        market_memo_config=market_memo_config,
        advanced_memo_config=advanced_memo_config,
        scenario_config=RegimeScenarioSimulationConfig(
            horizon=scenario_horizon,
            n_simulations=scenario_simulations,
            random_state=random_state,
        ),
        rolling_factor_window=rolling_factor_window,
    )

    return export_advanced_research_package(
        inputs=advanced_result.advanced_inputs,
        output_dir=output_dir,
        config=advanced_memo_config,
        overwrite=overwrite,
    )


def format_advanced_export_result(
    result: AdvancedResearchExportResult,
) -> str:
    """Format an advanced research export result for CLI output."""
    lines = [
        "Advanced research package exported successfully.",
        f"Output directory: {result.output_dir}",
        f"Memo: {result.memo_path}",
    ]

    if result.exported_table_paths:
        lines.append("Tables:")

        for name, path in sorted(result.exported_table_paths.items()):
            lines.append(f"- {name}: {path}")

    return "\n".join(lines)


def _read_price_data(path: str | Path) -> pd.DataFrame:
    frame = _read_csv(path, "price data")

    _require_columns(
        frame=frame,
        required_columns={"date", "ticker", "adjusted_close"},
        label="Price data",
    )

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["adjusted_close"] = pd.to_numeric(
        frame["adjusted_close"],
        errors="coerce",
    )

    if frame["date"].isna().any():
        raise AdvancedResearchCliExportError("Price data contains invalid dates")

    if frame["ticker"].str.len().eq(0).any():
        raise AdvancedResearchCliExportError("Price data contains empty tickers")

    if frame["adjusted_close"].isna().any():
        raise AdvancedResearchCliExportError(
            "Price data adjusted_close must be numeric and complete"
        )

    return frame


def _read_static_weights(path: str | Path) -> dict[str, float]:
    frame = _read_csv(path, "static weights")

    _require_columns(
        frame=frame,
        required_columns={"ticker", "weight"},
        label="Static weights",
    )

    weights = _read_weight_mapping(frame)

    _validate_weight_sum(weights, "Static weights")

    return weights


def _read_regime_policy(
    path: str | Path,
    static_weights: dict[str, float],
) -> dict[int, dict[str, float]]:
    frame = _read_csv(path, "regime policy")

    _require_columns(
        frame=frame,
        required_columns={"regime", "ticker", "weight"},
        label="Regime policy",
    )

    policy: dict[int, dict[str, float]] = {}

    for _, row in frame.iterrows():
        regime = _to_int(row["regime"], "regime")
        ticker = str(row["ticker"]).strip().upper()
        weight = _to_float(row["weight"], "regime policy weight")

        if not ticker:
            raise AdvancedResearchCliExportError(
                "Regime policy contains an empty ticker"
            )

        policy.setdefault(regime, {})[ticker] = weight

    if not policy:
        raise AdvancedResearchCliExportError("Regime policy cannot be empty")

    expected_tickers = set(static_weights)

    for regime, weights in policy.items():
        if set(weights) != expected_tickers:
            raise AdvancedResearchCliExportError(
                f"Regime {regime} policy tickers must match static weights"
            )

        _validate_weight_sum(weights, f"Regime {regime} policy")

    return policy


def _read_stress_periods(path: str | Path) -> list[StressPeriod]:
    frame = _read_csv(path, "stress periods")

    _require_columns(
        frame=frame,
        required_columns={"name", "start_date", "end_date"},
        label="Stress periods",
    )

    periods = [
        StressPeriod(
            name=str(row["name"]).strip(),
            start_date=str(row["start_date"]),
            end_date=str(row["end_date"]),
        )
        for _, row in frame.iterrows()
    ]

    if not periods:
        raise AdvancedResearchCliExportError("Stress periods cannot be empty")

    return periods


def _read_factor_returns(path: str | Path) -> pd.DataFrame:
    frame = _read_csv(path, "factor returns")

    _require_columns(
        frame=frame,
        required_columns={"date"},
        label="Factor returns",
    )

    if len(frame.columns) < 2:
        raise AdvancedResearchCliExportError(
            "Factor returns must include at least one factor column"
        )

    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    factor_frame = frame.drop(columns=["date"]).copy()
    factor_frame = factor_frame.apply(pd.to_numeric, errors="coerce")
    factor_frame.index = dates

    if factor_frame.isna().any().any():
        raise AdvancedResearchCliExportError(
            "Factor returns must be numeric and complete"
        )

    return factor_frame


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    csv_path = Path(path).expanduser()

    if not csv_path.exists():
        raise AdvancedResearchCliExportError(f"Missing {label} file: {csv_path}")

    if not csv_path.is_file():
        raise AdvancedResearchCliExportError(
            f"{label.capitalize()} path is not a file: {csv_path}"
        )

    return pd.read_csv(csv_path)


def _require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    label: str,
) -> None:
    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise AdvancedResearchCliExportError(f"{label} missing column(s): {missing}")


def _read_weight_mapping(frame: pd.DataFrame) -> dict[str, float]:
    weights: dict[str, float] = {}

    for _, row in frame.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        weight = _to_float(row["weight"], "weight")

        if not ticker:
            raise AdvancedResearchCliExportError("Weights contain an empty ticker")

        weights[ticker] = weight

    if not weights:
        raise AdvancedResearchCliExportError("Weights cannot be empty")

    return weights


def _build_static_regime_policy(
    static_weights: dict[str, float],
    n_regimes: int,
) -> dict[int, dict[str, float]]:
    return {regime: dict(static_weights) for regime in range(n_regimes)}


def _validate_weight_sum(weights: dict[str, float], label: str) -> None:
    weight_sum = sum(weights.values())

    if abs(weight_sum - 1.0) >= 1e-8:
        raise AdvancedResearchCliExportError(
            f"{label} must sum to 1.0; got {weight_sum:.6f}"
        )


def _validate_positive_int(value: int, label: str) -> None:
    if value <= 0:
        raise AdvancedResearchCliExportError(f"{label} must be positive")


def _to_float(value: object, label: str) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise AdvancedResearchCliExportError(f"Expected numeric {label}")

    return float(numeric)


def _to_int(value: object, label: str) -> int:
    numeric = _to_float(value, label)

    if not numeric.is_integer():
        raise AdvancedResearchCliExportError(f"Expected integer {label}")

    return int(numeric)
