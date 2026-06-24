from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from regime_risk_engine.research.advanced_memo import (
    AdvancedResearchMemoConfig,
    AdvancedResearchMemoInputs,
    build_advanced_research_memo,
)
from regime_risk_engine.research.rolling_factor_exposure import (
    summarize_rolling_factor_exposures,
    write_rolling_factor_exposures_csv,
)


class AdvancedResearchExportError(ValueError):
    """Raised when advanced research export cannot be completed."""


@dataclass(frozen=True, slots=True)
class AdvancedResearchExportResult:
    """Paths created by the advanced research export package."""

    output_dir: Path
    memo_path: Path
    exported_table_paths: dict[str, Path]


ADVANCED_MEMO_FILENAME = "advanced_research_memo.md"


def export_advanced_research_package(
    inputs: AdvancedResearchMemoInputs,
    output_dir: str | Path,
    config: AdvancedResearchMemoConfig | None = None,
    overwrite: bool = True,
) -> AdvancedResearchExportResult:
    """Export an advanced research memo and supporting tables."""
    clean_output_dir = _prepare_output_dir(output_dir)

    memo_path = clean_output_dir / ADVANCED_MEMO_FILENAME
    table_paths = _build_table_paths(
        inputs=inputs,
        output_dir=clean_output_dir,
    )

    _validate_overwrite_policy(
        memo_path=memo_path,
        table_paths=table_paths,
        overwrite=overwrite,
    )

    memo = build_advanced_research_memo(
        inputs=inputs,
        config=config,
    )
    memo_path.write_text(memo, encoding="utf-8")

    exported_table_paths = _export_tables(
        inputs=inputs,
        table_paths=table_paths,
    )

    return AdvancedResearchExportResult(
        output_dir=clean_output_dir,
        memo_path=memo_path,
        exported_table_paths=exported_table_paths,
    )


def _prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser().resolve()

    if path.exists() and not path.is_dir():
        raise AdvancedResearchExportError(
            f"Output path exists and is not a directory: {path}"
        )

    path.mkdir(parents=True, exist_ok=True)

    return path


def _build_table_paths(
    inputs: AdvancedResearchMemoInputs,
    output_dir: Path,
) -> dict[str, Path]:
    table_paths: dict[str, Path] = {}

    if inputs.regime_intelligence is not None:
        table_paths["regime_intelligence_profile"] = (
            output_dir / "regime_intelligence_profile.csv"
        )

    if inputs.regime_transitions is not None:
        table_paths["regime_transition_counts"] = (
            output_dir / "regime_transition_counts.csv"
        )
        table_paths["regime_transition_probabilities"] = (
            output_dir / "regime_transition_probabilities.csv"
        )
        table_paths["regime_persistence"] = output_dir / "regime_persistence.csv"
        table_paths["regime_durations"] = output_dir / "regime_durations.csv"
        table_paths["most_likely_next_regime"] = (
            output_dir / "most_likely_next_regime.csv"
        )

    if inputs.stress_test is not None:
        table_paths["stress_test_summary"] = output_dir / "stress_test_summary.csv"

    if inputs.attribution is not None:
        table_paths["asset_attribution"] = output_dir / "asset_attribution.csv"

        if inputs.attribution.regime_attribution is not None:
            table_paths["regime_attribution"] = output_dir / "regime_attribution.csv"

    if inputs.factor_exposure is not None:
        table_paths["factor_exposure"] = output_dir / "factor_exposure.csv"
        table_paths["dominant_factor_by_strategy"] = (
            output_dir / "dominant_factor_by_strategy.csv"
        )

        if inputs.factor_exposure.regime_exposure_table is not None:
            table_paths["regime_factor_exposure"] = (
                output_dir / "regime_factor_exposure.csv"
            )

    if inputs.rolling_factor_exposure is not None:
        table_paths["rolling_factor_exposures"] = (
            output_dir / "rolling_factor_exposures.csv"
        )
        table_paths["rolling_factor_exposure_summary"] = (
            output_dir / "rolling_factor_exposure_summary.csv"
        )

    if inputs.factor_significance is not None:
        table_paths["factor_significance"] = output_dir / "factor_significance.csv"

    if inputs.scenario_simulation is not None:
        table_paths["scenario_terminal_summary"] = (
            output_dir / "scenario_terminal_summary.csv"
        )
        table_paths["scenario_regime_usage"] = output_dir / "scenario_regime_usage.csv"
        table_paths["scenario_simulated_paths"] = (
            output_dir / "scenario_simulated_paths.csv"
        )
        table_paths["scenario_transition_probabilities"] = (
            output_dir / "scenario_transition_probabilities.csv"
        )

    return table_paths


def _validate_overwrite_policy(
    memo_path: Path,
    table_paths: dict[str, Path],
    overwrite: bool,
) -> None:
    if overwrite:
        return

    existing_paths = [
        path for path in [memo_path, *table_paths.values()] if path.exists()
    ]

    if existing_paths:
        existing = ", ".join(path.name for path in existing_paths)
        raise AdvancedResearchExportError(
            f"Export file(s) already exist and overwrite=False: {existing}"
        )


def _export_tables(
    inputs: AdvancedResearchMemoInputs,
    table_paths: dict[str, Path],
) -> dict[str, Path]:
    exported: dict[str, Path] = {}

    if inputs.regime_intelligence is not None:
        _write_table(
            inputs.regime_intelligence.profile_table,
            table_paths["regime_intelligence_profile"],
        )
        exported["regime_intelligence_profile"] = table_paths[
            "regime_intelligence_profile"
        ]

    if inputs.regime_transitions is not None:
        _write_table(
            inputs.regime_transitions.transition_counts,
            table_paths["regime_transition_counts"],
        )
        exported["regime_transition_counts"] = table_paths["regime_transition_counts"]

        _write_table(
            inputs.regime_transitions.transition_probabilities,
            table_paths["regime_transition_probabilities"],
        )
        exported["regime_transition_probabilities"] = table_paths[
            "regime_transition_probabilities"
        ]

        _write_table(
            inputs.regime_transitions.regime_persistence,
            table_paths["regime_persistence"],
        )
        exported["regime_persistence"] = table_paths["regime_persistence"]

        _write_table(
            inputs.regime_transitions.regime_durations,
            table_paths["regime_durations"],
        )
        exported["regime_durations"] = table_paths["regime_durations"]

        _write_table(
            inputs.regime_transitions.most_likely_next_regime,
            table_paths["most_likely_next_regime"],
        )
        exported["most_likely_next_regime"] = table_paths["most_likely_next_regime"]

    if inputs.stress_test is not None:
        _write_table(
            inputs.stress_test.summary_table,
            table_paths["stress_test_summary"],
        )
        exported["stress_test_summary"] = table_paths["stress_test_summary"]

    if inputs.attribution is not None:
        _write_table(
            inputs.attribution.asset_attribution,
            table_paths["asset_attribution"],
        )
        exported["asset_attribution"] = table_paths["asset_attribution"]

        if inputs.attribution.regime_attribution is not None:
            _write_table(
                inputs.attribution.regime_attribution,
                table_paths["regime_attribution"],
            )
            exported["regime_attribution"] = table_paths["regime_attribution"]

    if inputs.factor_exposure is not None:
        _write_table(
            inputs.factor_exposure.exposure_table,
            table_paths["factor_exposure"],
        )
        exported["factor_exposure"] = table_paths["factor_exposure"]

        _write_table(
            inputs.factor_exposure.dominant_factor_by_strategy,
            table_paths["dominant_factor_by_strategy"],
        )
        exported["dominant_factor_by_strategy"] = table_paths[
            "dominant_factor_by_strategy"
        ]

        if inputs.factor_exposure.regime_exposure_table is not None:
            _write_table(
                inputs.factor_exposure.regime_exposure_table,
                table_paths["regime_factor_exposure"],
            )
            exported["regime_factor_exposure"] = table_paths["regime_factor_exposure"]

    if inputs.rolling_factor_exposure is not None:
        write_rolling_factor_exposures_csv(
            table_paths["rolling_factor_exposures"],
            inputs.rolling_factor_exposure,
        )
        exported["rolling_factor_exposures"] = table_paths["rolling_factor_exposures"]

        _write_table(
            summarize_rolling_factor_exposures(inputs.rolling_factor_exposure),
            table_paths["rolling_factor_exposure_summary"],
        )
        exported["rolling_factor_exposure_summary"] = table_paths[
            "rolling_factor_exposure_summary"
        ]

    if inputs.factor_significance is not None:
        _write_table(
            inputs.factor_significance.significance_table,
            table_paths["factor_significance"],
        )
        exported["factor_significance"] = table_paths["factor_significance"]

    if inputs.scenario_simulation is not None:
        _write_table(
            inputs.scenario_simulation.terminal_summary,
            table_paths["scenario_terminal_summary"],
        )
        exported["scenario_terminal_summary"] = table_paths["scenario_terminal_summary"]

        _write_table(
            inputs.scenario_simulation.regime_usage,
            table_paths["scenario_regime_usage"],
        )
        exported["scenario_regime_usage"] = table_paths["scenario_regime_usage"]

        _write_table(
            inputs.scenario_simulation.simulated_paths,
            table_paths["scenario_simulated_paths"],
        )
        exported["scenario_simulated_paths"] = table_paths["scenario_simulated_paths"]

        _write_table(
            inputs.scenario_simulation.transition_probabilities,
            table_paths["scenario_transition_probabilities"],
        )
        exported["scenario_transition_probabilities"] = table_paths[
            "scenario_transition_probabilities"
        ]

    return exported


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise AdvancedResearchExportError(f"Cannot export empty table: {path.name}")

    frame.to_csv(path, index=True, index_label="index")
