import pandas as pd
import pytest

from regime_risk_engine.research.scenario_simulation import (
    RegimeScenarioSimulationConfig,
    RegimeScenarioSimulationError,
    RegimeScenarioSimulationResult,
    simulate_regime_strategy_scenarios,
)


def make_return_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    rows = []

    for index, _date in enumerate(dates):
        if index < 30:
            rows.append(
                {
                    "static": 0.002,
                    "dynamic": 0.003,
                    "regime": 0,
                }
            )
        elif index < 60:
            rows.append(
                {
                    "static": -0.006,
                    "dynamic": -0.002,
                    "regime": 1,
                }
            )
        else:
            rows.append(
                {
                    "static": 0.0005,
                    "dynamic": 0.0015,
                    "regime": 2,
                }
            )

    return pd.DataFrame(rows, index=dates)


def make_config() -> RegimeScenarioSimulationConfig:
    return RegimeScenarioSimulationConfig(
        horizon=10,
        n_simulations=50,
        random_state=7,
        initial_regime=0,
    )


def test_simulate_regime_strategy_scenarios() -> None:
    result = simulate_regime_strategy_scenarios(
        return_frame=make_return_frame(),
        config=make_config(),
    )

    assert isinstance(result, RegimeScenarioSimulationResult)
    assert len(result.simulated_paths) == 500
    assert len(result.terminal_summary) == 2
    assert len(result.transition_probabilities) == 3
    assert len(result.regime_usage) == 3
    assert "Regime scenario simulation generated" in result.narrative


def test_terminal_summary_contains_risk_metrics() -> None:
    result = simulate_regime_strategy_scenarios(
        return_frame=make_return_frame(),
        config=make_config(),
    )

    expected_columns = {
        "strategy",
        "mean_terminal_return",
        "median_terminal_return",
        "terminal_return_volatility",
        "probability_of_loss",
        "var_95",
        "cvar_95",
        "best_terminal_return",
        "worst_terminal_return",
    }

    assert expected_columns.issubset(result.terminal_summary.columns)


def test_transition_probabilities_sum_to_one() -> None:
    result = simulate_regime_strategy_scenarios(
        return_frame=make_return_frame(),
        config=make_config(),
    )

    row_sums = result.transition_probabilities.sum(axis=1)

    assert all(abs(row_sum - 1.0) < 1e-12 for row_sum in row_sums)


def test_regime_usage_sums_to_one() -> None:
    result = simulate_regime_strategy_scenarios(
        return_frame=make_return_frame(),
        config=make_config(),
    )

    assert abs(result.regime_usage["simulated_frequency"].sum() - 1.0) < 1e-12


def test_scenario_simulation_accepts_custom_strategy_columns() -> None:
    return_frame = make_return_frame().rename(
        columns={
            "static": "benchmark",
            "dynamic": "candidate",
        }
    )

    result = simulate_regime_strategy_scenarios(
        return_frame=return_frame,
        strategy_columns=("benchmark", "candidate"),
        config=make_config(),
    )

    assert set(result.terminal_summary["strategy"]) == {"benchmark", "candidate"}


def test_scenario_simulation_rejects_empty_return_frame() -> None:
    with pytest.raises(RegimeScenarioSimulationError, match="Return frame"):
        simulate_regime_strategy_scenarios(
            return_frame=pd.DataFrame(),
            config=make_config(),
        )


def test_scenario_simulation_rejects_missing_column() -> None:
    return_frame = make_return_frame().drop(columns=["dynamic"])

    with pytest.raises(RegimeScenarioSimulationError, match="Missing required"):
        simulate_regime_strategy_scenarios(
            return_frame=return_frame,
            config=make_config(),
        )


def test_scenario_simulation_rejects_non_numeric_returns() -> None:
    return_frame = make_return_frame()
    return_frame["static"] = return_frame["static"].astype(object)
    return_frame.loc[return_frame.index[0], "static"] = "bad"

    with pytest.raises(RegimeScenarioSimulationError, match="numeric"):
        simulate_regime_strategy_scenarios(
            return_frame=return_frame,
            config=make_config(),
        )


def test_scenario_simulation_rejects_single_regime() -> None:
    return_frame = make_return_frame()
    return_frame["regime"] = 0

    with pytest.raises(RegimeScenarioSimulationError, match="At least two regimes"):
        simulate_regime_strategy_scenarios(
            return_frame=return_frame,
            config=make_config(),
        )


def test_scenario_simulation_rejects_unknown_initial_regime() -> None:
    with pytest.raises(RegimeScenarioSimulationError, match="Initial regime"):
        simulate_regime_strategy_scenarios(
            return_frame=make_return_frame(),
            config=RegimeScenarioSimulationConfig(
                horizon=10,
                n_simulations=50,
                random_state=7,
                initial_regime=99,
            ),
        )


def test_scenario_simulation_rejects_bad_horizon() -> None:
    with pytest.raises(RegimeScenarioSimulationError, match="horizon"):
        simulate_regime_strategy_scenarios(
            return_frame=make_return_frame(),
            config=RegimeScenarioSimulationConfig(horizon=0),
        )
