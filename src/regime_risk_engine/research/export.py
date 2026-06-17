from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from regime_risk_engine.research.market_workflow import MarketResearchWorkflowResult
from regime_risk_engine.research.memo import (
    MarketResearchMemoConfig,
    build_market_research_memo,
)


class MarketResearchExportError(ValueError):
    """Raised when market research export cannot be completed."""


@dataclass(frozen=True, slots=True)
class MarketResearchExportResult:
    """File paths created by the market research export package."""

    output_dir: Path
    memo_path: Path
    strategy_metric_table_path: Path
    metric_delta_table_path: Path
    regime_metric_table_path: Path
    regime_summary_table_path: Path
    dynamic_target_weights_path: Path
    dynamic_applied_weights_path: Path
    regime_labels_path: Path


MEMO_FILENAME: Final[str] = "investment_memo.md"
STRATEGY_METRIC_TABLE_FILENAME: Final[str] = "strategy_metric_table.csv"
METRIC_DELTA_TABLE_FILENAME: Final[str] = "metric_delta_table.csv"
REGIME_METRIC_TABLE_FILENAME: Final[str] = "regime_metric_table.csv"
REGIME_SUMMARY_TABLE_FILENAME: Final[str] = "regime_summary_table.csv"
DYNAMIC_TARGET_WEIGHTS_FILENAME: Final[str] = "dynamic_target_weights.csv"
DYNAMIC_APPLIED_WEIGHTS_FILENAME: Final[str] = "dynamic_applied_weights.csv"
REGIME_LABELS_FILENAME: Final[str] = "regime_labels.csv"


def export_market_research_result(
    result: MarketResearchWorkflowResult,
    output_dir: str | Path,
    memo_config: MarketResearchMemoConfig | None = None,
    overwrite: bool = True,
) -> MarketResearchExportResult:
    """Export market research result tables and memo to a directory."""
    clean_output_dir = _prepare_output_dir(output_dir)

    export_result = MarketResearchExportResult(
        output_dir=clean_output_dir,
        memo_path=clean_output_dir / MEMO_FILENAME,
        strategy_metric_table_path=clean_output_dir / STRATEGY_METRIC_TABLE_FILENAME,
        metric_delta_table_path=clean_output_dir / METRIC_DELTA_TABLE_FILENAME,
        regime_metric_table_path=clean_output_dir / REGIME_METRIC_TABLE_FILENAME,
        regime_summary_table_path=clean_output_dir / REGIME_SUMMARY_TABLE_FILENAME,
        dynamic_target_weights_path=clean_output_dir / DYNAMIC_TARGET_WEIGHTS_FILENAME,
        dynamic_applied_weights_path=clean_output_dir
        / DYNAMIC_APPLIED_WEIGHTS_FILENAME,
        regime_labels_path=clean_output_dir / REGIME_LABELS_FILENAME,
    )

    _validate_overwrite_policy(export_result, overwrite=overwrite)

    memo = build_market_research_memo(
        result=result,
        config=memo_config,
    )

    export_result.memo_path.write_text(memo, encoding="utf-8")

    _write_csv(
        result.research_result.strategy_metric_table,
        export_result.strategy_metric_table_path,
    )
    _write_csv(
        result.research_result.metric_delta_table,
        export_result.metric_delta_table_path,
    )
    _write_csv(
        result.research_result.regime_metric_table,
        export_result.regime_metric_table_path,
    )
    _write_csv(
        result.research_result.regime_summary.regime_summary,
        export_result.regime_summary_table_path,
    )
    _write_csv(
        result.dynamic_target_weight_frame,
        export_result.dynamic_target_weights_path,
    )
    _write_csv(
        result.dynamic_applied_weight_frame,
        export_result.dynamic_applied_weights_path,
    )
    _write_csv(
        result.regime_labels.rename("regime").to_frame(),
        export_result.regime_labels_path,
    )

    return export_result


def _prepare_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir).expanduser().resolve()

    if path.exists() and not path.is_dir():
        raise MarketResearchExportError(
            f"Output path exists and is not a directory: {path}"
        )

    path.mkdir(parents=True, exist_ok=True)

    return path


def _validate_overwrite_policy(
    export_result: MarketResearchExportResult,
    overwrite: bool,
) -> None:
    if overwrite:
        return

    existing_paths = [path for path in _export_paths(export_result) if path.exists()]

    if existing_paths:
        existing = ", ".join(path.name for path in existing_paths)
        raise MarketResearchExportError(
            f"Export file(s) already exist and overwrite=False: {existing}"
        )


def _export_paths(export_result: MarketResearchExportResult) -> list[Path]:
    return [
        export_result.memo_path,
        export_result.strategy_metric_table_path,
        export_result.metric_delta_table_path,
        export_result.regime_metric_table_path,
        export_result.regime_summary_table_path,
        export_result.dynamic_target_weights_path,
        export_result.dynamic_applied_weights_path,
        export_result.regime_labels_path,
    ]


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        raise MarketResearchExportError(f"Cannot export empty table: {path.name}")

    frame.to_csv(path, index=True, index_label="index")
