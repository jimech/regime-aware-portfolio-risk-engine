from dataclasses import dataclass

import numpy as np
import pandas as pd


class RegimeScenarioSimulationError(ValueError):
    """Raised when regime scenario simulation cannot be completed."""


@dataclass(frozen=True, slots=True)
class RegimeScenarioSimulationConfig:
    """Configuration for regime scenario simulation."""

    horizon: int = 21
    n_simulations: int = 1_000
    random_state: int = 42
    initial_regime: int | None = None


@dataclass(frozen=True, slots=True)
class RegimeScenarioSimulationResult:
    """Regime scenario simulation result."""

    simulated_paths: pd.DataFrame
    terminal_summary: pd.DataFrame
    transition_probabilities: pd.DataFrame
    regime_usage: pd.DataFrame
    narrative: str


def simulate_regime_strategy_scenarios(
    return_frame: pd.DataFrame,
    regime_column: str = "regime",
    strategy_columns: tuple[str, ...] = ("static", "dynamic"),
    config: RegimeScenarioSimulationConfig | None = None,
) -> RegimeScenarioSimulationResult:
    """Simulate future strategy outcomes using regime transition behavior."""
    simulation_config = config or RegimeScenarioSimulationConfig()

    _validate_config(simulation_config)
    _validate_inputs(
        return_frame=return_frame,
        regime_column=regime_column,
        strategy_columns=strategy_columns,
    )

    clean_frame = _prepare_return_frame(
        return_frame=return_frame,
        regime_column=regime_column,
        strategy_columns=strategy_columns,
    )
    regimes = sorted(int(regime) for regime in clean_frame["regime"].unique())

    transition_probabilities = _build_transition_probability_matrix(
        regime_labels=clean_frame["regime"],
        regimes=regimes,
    )
    regime_observation_positions = _build_regime_observation_positions(clean_frame)
    initial_regime = _resolve_initial_regime(
        clean_frame=clean_frame,
        config=simulation_config,
        regimes=regimes,
    )

    simulated_paths = _simulate_paths(
        clean_frame=clean_frame,
        regimes=regimes,
        transition_probabilities=transition_probabilities,
        regime_observation_positions=regime_observation_positions,
        initial_regime=initial_regime,
        strategy_columns=strategy_columns,
        config=simulation_config,
    )
    terminal_summary = _build_terminal_summary(
        simulated_paths=simulated_paths,
        strategy_columns=strategy_columns,
    )
    regime_usage = _build_regime_usage_table(simulated_paths)
    narrative = _build_narrative(
        terminal_summary=terminal_summary,
        regime_usage=regime_usage,
        strategy_columns=strategy_columns,
        horizon=simulation_config.horizon,
        n_simulations=simulation_config.n_simulations,
    )

    return RegimeScenarioSimulationResult(
        simulated_paths=simulated_paths,
        terminal_summary=terminal_summary,
        transition_probabilities=transition_probabilities,
        regime_usage=regime_usage,
        narrative=narrative,
    )


def _validate_config(config: RegimeScenarioSimulationConfig) -> None:
    if config.horizon <= 0:
        raise RegimeScenarioSimulationError("Simulation horizon must be positive")

    if config.n_simulations <= 0:
        raise RegimeScenarioSimulationError("Number of simulations must be positive")


def _validate_inputs(
    return_frame: pd.DataFrame,
    regime_column: str,
    strategy_columns: tuple[str, ...],
) -> None:
    if return_frame.empty:
        raise RegimeScenarioSimulationError("Return frame cannot be empty")

    if not isinstance(return_frame.index, pd.DatetimeIndex):
        raise RegimeScenarioSimulationError(
            "Return frame index must be a DatetimeIndex"
        )

    if not regime_column.strip():
        raise RegimeScenarioSimulationError("Regime column must be non-empty")

    if not strategy_columns:
        raise RegimeScenarioSimulationError("At least one strategy column is required")

    missing_columns = {regime_column, *strategy_columns}.difference(
        return_frame.columns
    )

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RegimeScenarioSimulationError(f"Missing required column(s): {missing}")

    strategy_returns = return_frame[list(strategy_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if strategy_returns.isna().any().any():
        raise RegimeScenarioSimulationError(
            "Strategy returns must be numeric and complete"
        )

    regimes = pd.to_numeric(return_frame[regime_column], errors="coerce")

    if regimes.isna().any():
        raise RegimeScenarioSimulationError(
            "Regime values must be numeric and complete"
        )

    if regimes.nunique() < 2:
        raise RegimeScenarioSimulationError("At least two regimes are required")


def _prepare_return_frame(
    return_frame: pd.DataFrame,
    regime_column: str,
    strategy_columns: tuple[str, ...],
) -> pd.DataFrame:
    clean_frame = return_frame[[*strategy_columns, regime_column]].copy()

    for column in strategy_columns:
        clean_frame[column] = pd.to_numeric(clean_frame[column], errors="coerce")

    clean_frame["regime"] = pd.to_numeric(
        clean_frame[regime_column],
        errors="coerce",
    ).astype(int)

    if regime_column != "regime":
        clean_frame = clean_frame.drop(columns=[regime_column])

    return clean_frame.sort_index()


def _build_transition_probability_matrix(
    regime_labels: pd.Series,
    regimes: list[int],
) -> pd.DataFrame:
    transition_counts = pd.DataFrame(
        0,
        index=pd.Index(regimes, name="current_regime"),
        columns=pd.Index(regimes, name="next_regime"),
        dtype=int,
    )

    current_values = regime_labels.iloc[:-1].to_numpy(dtype=int)
    next_values = regime_labels.iloc[1:].to_numpy(dtype=int)

    for current_regime, next_regime in zip(
        current_values,
        next_values,
        strict=True,
    ):
        transition_counts.loc[int(current_regime), int(next_regime)] += 1

    transition_probabilities = transition_counts.astype(float)

    for regime in regimes:
        row_sum = float(transition_counts.loc[regime].sum())

        if row_sum == 0.0:
            transition_probabilities.loc[regime] = 0.0
            transition_probabilities.loc[regime, regime] = 1.0
        else:
            transition_probabilities.loc[regime] = (
                transition_counts.loc[regime].astype(float) / row_sum
            )

    return transition_probabilities


def _build_regime_observation_positions(
    clean_frame: pd.DataFrame,
) -> dict[int, list[int]]:
    observation_positions: dict[int, list[int]] = {}

    reset_frame = clean_frame.reset_index(drop=True)

    for regime, regime_frame in reset_frame.groupby("regime"):
        observation_positions[int(regime)] = [
            int(position) for position in regime_frame.index.to_list()
        ]

    return observation_positions


def _resolve_initial_regime(
    clean_frame: pd.DataFrame,
    config: RegimeScenarioSimulationConfig,
    regimes: list[int],
) -> int:
    if config.initial_regime is None:
        return int(clean_frame["regime"].iloc[-1])

    if config.initial_regime not in regimes:
        raise RegimeScenarioSimulationError(
            f"Initial regime is not present in historical data: {config.initial_regime}"
        )

    return int(config.initial_regime)


def _simulate_paths(
    clean_frame: pd.DataFrame,
    regimes: list[int],
    transition_probabilities: pd.DataFrame,
    regime_observation_positions: dict[int, list[int]],
    initial_regime: int,
    strategy_columns: tuple[str, ...],
    config: RegimeScenarioSimulationConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)
    records: list[dict[str, float | int]] = []

    for simulation_id in range(config.n_simulations):
        current_regime = initial_regime
        wealth = {strategy: 1.0 for strategy in strategy_columns}

        for step in range(1, config.horizon + 1):
            current_regime = _sample_next_regime(
                current_regime=current_regime,
                regimes=regimes,
                transition_probabilities=transition_probabilities,
                rng=rng,
            )
            observation_position = _sample_observation_position(
                observation_positions=regime_observation_positions[current_regime],
                rng=rng,
            )
            observation = clean_frame.iloc[observation_position]

            record: dict[str, float | int] = {
                "simulation_id": simulation_id,
                "step": step,
                "regime": current_regime,
            }

            for strategy in strategy_columns:
                strategy_return = _to_float(
                    observation[strategy],
                    f"{strategy} simulated return",
                )
                wealth[strategy] *= 1.0 + strategy_return

                record[f"{strategy}_return"] = strategy_return
                record[f"{strategy}_wealth"] = wealth[strategy]

            records.append(record)

    return pd.DataFrame(records)


def _sample_next_regime(
    current_regime: int,
    regimes: list[int],
    transition_probabilities: pd.DataFrame,
    rng: np.random.Generator,
) -> int:
    probabilities = np.asarray(
        transition_probabilities.loc[current_regime].to_numpy(dtype=float),
        dtype=float,
    )
    probability_sum = float(probabilities.sum())

    if probability_sum <= 0.0:
        return current_regime

    normalized_probabilities = probabilities / probability_sum
    selected_regime = rng.choice(
        np.asarray(regimes, dtype=int),
        p=normalized_probabilities,
    )

    return int(selected_regime)


def _sample_observation_position(
    observation_positions: list[int],
    rng: np.random.Generator,
) -> int:
    if not observation_positions:
        raise RegimeScenarioSimulationError(
            "Cannot sample from regime with no observations"
        )

    selected_position = rng.choice(np.asarray(observation_positions, dtype=int))

    return int(selected_position)


def _build_terminal_summary(
    simulated_paths: pd.DataFrame,
    strategy_columns: tuple[str, ...],
) -> pd.DataFrame:
    if simulated_paths.empty:
        raise RegimeScenarioSimulationError("Simulated paths cannot be empty")

    rows = []

    for strategy in strategy_columns:
        wealth_column = f"{strategy}_wealth"
        terminal_returns = (
            simulated_paths.groupby("simulation_id")[wealth_column].last() - 1.0
        )

        var_95 = float(terminal_returns.quantile(0.05))
        tail_returns = terminal_returns[terminal_returns <= var_95]

        if tail_returns.empty:
            cvar_95 = var_95
        else:
            cvar_95 = float(tail_returns.mean())

        rows.append(
            {
                "strategy": strategy,
                "mean_terminal_return": float(terminal_returns.mean()),
                "median_terminal_return": float(terminal_returns.median()),
                "terminal_return_volatility": float(terminal_returns.std(ddof=1)),
                "probability_of_loss": float((terminal_returns < 0.0).mean()),
                "var_95": var_95,
                "cvar_95": cvar_95,
                "best_terminal_return": float(terminal_returns.max()),
                "worst_terminal_return": float(terminal_returns.min()),
            }
        )

    return pd.DataFrame(rows)


def _build_regime_usage_table(simulated_paths: pd.DataFrame) -> pd.DataFrame:
    regime_counts = (
        simulated_paths["regime"].astype(int).value_counts(normalize=True).sort_index()
    )

    rows = [
        {
            "regime": int(regime),
            "simulated_frequency": float(frequency),
        }
        for regime, frequency in regime_counts.items()
    ]

    return pd.DataFrame(rows)


def _build_narrative(
    terminal_summary: pd.DataFrame,
    regime_usage: pd.DataFrame,
    strategy_columns: tuple[str, ...],
    horizon: int,
    n_simulations: int,
) -> str:
    narrative = (
        f"Regime scenario simulation generated {n_simulations} forward paths "
        f"over a {horizon}-period horizon using historical regime transition "
        "behavior and regime-conditioned return sampling."
    )

    if len(strategy_columns) >= 2:
        benchmark_strategy = strategy_columns[0]
        candidate_strategy = strategy_columns[1]

        benchmark_mean = _lookup_terminal_metric(
            terminal_summary=terminal_summary,
            strategy=benchmark_strategy,
            metric="mean_terminal_return",
        )
        candidate_mean = _lookup_terminal_metric(
            terminal_summary=terminal_summary,
            strategy=candidate_strategy,
            metric="mean_terminal_return",
        )
        benchmark_cvar = _lookup_terminal_metric(
            terminal_summary=terminal_summary,
            strategy=benchmark_strategy,
            metric="cvar_95",
        )
        candidate_cvar = _lookup_terminal_metric(
            terminal_summary=terminal_summary,
            strategy=candidate_strategy,
            metric="cvar_95",
        )

        narrative += (
            f" The simulated mean terminal return difference for "
            f"{candidate_strategy} versus {benchmark_strategy} was "
            f"{candidate_mean - benchmark_mean:.4f}."
        )

        narrative += (
            f" The simulated 95% CVaR difference was "
            f"{candidate_cvar - benchmark_cvar:.4f}, where a higher value "
            "indicates a less severe tail outcome."
        )

    if not regime_usage.empty:
        most_common_regime = int(
            regime_usage.sort_values(
                "simulated_frequency",
                ascending=False,
            ).iloc[0]["regime"]
        )
        narrative += (
            f" The most frequently simulated regime was regime {most_common_regime}."
        )

    return narrative


def _lookup_terminal_metric(
    terminal_summary: pd.DataFrame,
    strategy: str,
    metric: str,
) -> float:
    row = terminal_summary[terminal_summary["strategy"] == strategy]

    if row.empty:
        raise RegimeScenarioSimulationError(
            f"Missing terminal metric for strategy: {strategy}"
        )

    return _to_float(row.iloc[0][metric], f"{strategy} {metric}")


def _to_float(value: object, context: str) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if pd.isna(numeric):
        raise RegimeScenarioSimulationError(f"Expected numeric value for {context}")

    return float(numeric)
