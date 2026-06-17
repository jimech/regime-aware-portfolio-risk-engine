import pandas as pd
import pytest

from regime_risk_engine.research.market_workflow import (
    MarketResearchWorkflowConfig,
    run_market_research_workflow,
)
from regime_risk_engine.research.memo import (
    MarketResearchMemoConfig,
    MarketResearchMemoError,
    build_market_research_memo,
)


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


def test_build_market_research_memo() -> None:
    memo = build_market_research_memo(make_market_research_result())

    assert "# Regime-Aware Portfolio Research Memo" in memo
    assert "## Executive Summary" in memo
    assert "## Strategy Performance" in memo
    assert "## Regime-Level Findings" in memo
    assert "## Dynamic Allocation Profile" in memo
    assert "## Research Conclusion" in memo
    assert "## Limitations" in memo
    assert "dynamic regime-aware strategy" in memo


def test_build_market_research_memo_with_custom_config() -> None:
    memo = build_market_research_memo(
        result=make_market_research_result(),
        config=MarketResearchMemoConfig(
            title="Internal Portfolio Review",
            analyst="Jimena Chinchilla",
            include_limitations=False,
        ),
    )

    assert "# Internal Portfolio Review" in memo
    assert "**Analyst:** Jimena Chinchilla" in memo
    assert "## Limitations" not in memo


def test_market_research_memo_rejects_empty_title() -> None:
    with pytest.raises(MarketResearchMemoError, match="title"):
        build_market_research_memo(
            result=make_market_research_result(),
            config=MarketResearchMemoConfig(title=" "),
        )


def test_market_research_memo_rejects_empty_analyst() -> None:
    with pytest.raises(MarketResearchMemoError, match="Analyst"):
        build_market_research_memo(
            result=make_market_research_result(),
            config=MarketResearchMemoConfig(analyst=" "),
        )
