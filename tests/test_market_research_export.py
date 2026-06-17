import pandas as pd
import pytest

from regime_risk_engine.research.export import (
    MarketResearchExportError,
    MarketResearchExportResult,
    export_market_research_result,
)
from regime_risk_engine.research.market_workflow import (
    MarketResearchWorkflowConfig,
    run_market_research_workflow,
)
from regime_risk_engine.research.memo import MarketResearchMemoConfig


def make_price_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    rows = []

    spy_price = 100.0
    tlt_price = 100.0
    gld_price = 100.0

    for index, date in enumerate(dates):
        if index < 15:
            spy_return = 0.004
            tlt_return = 0.001
            gld_return = 0.0005
        elif index < 28:
            spy_return = -0.006
            tlt_return = 0.004
            gld_return = 0.003
        else:
            spy_return = 0.002
            tlt_return = -0.001
            gld_return = 0.004

        spy_price *= 1.0 + spy_return
        tlt_price *= 1.0 + tlt_return
        gld_price *= 1.0 + gld_return

        rows.extend(
            [
                {
                    "date": date,
                    "ticker": "SPY",
                    "adjusted_close": spy_price,
                },
                {
                    "date": date,
                    "ticker": "TLT",
                    "adjusted_close": tlt_price,
                },
                {
                    "date": date,
                    "ticker": "GLD",
                    "adjusted_close": gld_price,
                },
            ]
        )

    return pd.DataFrame(rows)


def make_static_weights() -> dict[str, float]:
    return {
        "SPY": 0.60,
        "TLT": 0.30,
        "GLD": 0.10,
    }


def make_regime_policy() -> dict[int, dict[str, float]]:
    return {
        0: {
            "SPY": 0.70,
            "TLT": 0.20,
            "GLD": 0.10,
        },
        1: {
            "SPY": 0.30,
            "TLT": 0.50,
            "GLD": 0.20,
        },
        2: {
            "SPY": 0.40,
            "TLT": 0.20,
            "GLD": 0.40,
        },
    }


def make_market_research_result():
    return run_market_research_workflow(
        price_data=make_price_data(),
        static_weights=make_static_weights(),
        regime_weight_policy=make_regime_policy(),
        config=MarketResearchWorkflowConfig(
            n_regimes=3,
            feature_window=5,
            transaction_cost_bps=1.0,
            random_state=7,
        ),
    )


def test_export_market_research_result(tmp_path) -> None:
    export_result = export_market_research_result(
        result=make_market_research_result(),
        output_dir=tmp_path / "research_export",
    )

    assert isinstance(export_result, MarketResearchExportResult)
    assert export_result.output_dir.exists()
    assert export_result.memo_path.exists()
    assert export_result.strategy_metric_table_path.exists()
    assert export_result.metric_delta_table_path.exists()
    assert export_result.regime_metric_table_path.exists()
    assert export_result.regime_summary_table_path.exists()
    assert export_result.dynamic_target_weights_path.exists()
    assert export_result.dynamic_applied_weights_path.exists()
    assert export_result.regime_labels_path.exists()

    memo = export_result.memo_path.read_text(encoding="utf-8")

    assert "# Regime-Aware Portfolio Research Memo" in memo
    assert "## Executive Summary" in memo


def test_export_market_research_result_with_custom_memo_config(tmp_path) -> None:
    export_result = export_market_research_result(
        result=make_market_research_result(),
        output_dir=tmp_path / "custom_export",
        memo_config=MarketResearchMemoConfig(
            title="Internal Allocation Review",
            analyst="Jimena Chinchilla",
            include_limitations=False,
        ),
    )

    memo = export_result.memo_path.read_text(encoding="utf-8")

    assert "# Internal Allocation Review" in memo
    assert "**Analyst:** Jimena Chinchilla" in memo
    assert "## Limitations" not in memo


def test_export_market_research_result_rejects_existing_files_without_overwrite(
    tmp_path,
) -> None:
    output_dir = tmp_path / "existing_export"

    export_market_research_result(
        result=make_market_research_result(),
        output_dir=output_dir,
    )

    with pytest.raises(MarketResearchExportError, match="overwrite=False"):
        export_market_research_result(
            result=make_market_research_result(),
            output_dir=output_dir,
            overwrite=False,
        )


def test_export_market_research_result_rejects_file_output_path(tmp_path) -> None:
    output_path = tmp_path / "not_a_directory.txt"
    output_path.write_text("already exists", encoding="utf-8")

    with pytest.raises(MarketResearchExportError, match="not a directory"):
        export_market_research_result(
            result=make_market_research_result(),
            output_dir=output_path,
        )
