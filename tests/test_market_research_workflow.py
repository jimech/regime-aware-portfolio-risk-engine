import pandas as pd
import pytest

from regime_risk_engine.research.market_workflow import (
    MarketResearchWorkflowConfig,
    MarketResearchWorkflowError,
    MarketResearchWorkflowResult,
    run_market_research_workflow,
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


def test_run_market_research_workflow() -> None:
    result = run_market_research_workflow(
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

    assert isinstance(result, MarketResearchWorkflowResult)
    assert not result.asset_returns.empty
    assert not result.feature_matrix.empty
    assert not result.regime_labels.empty
    assert set(result.regime_labels.unique()).issubset({0, 1, 2})
    assert not result.strategy_comparison.return_comparison.empty
    assert not result.strategy_comparison.metric_summary.empty
    assert not result.regime_evaluation.metric_summary.empty
    assert result.research_result.executive_summary
    assert result.research_result.strategy_summary.candidate_strategy == "dynamic"


def test_market_research_workflow_uses_lagged_dynamic_weights() -> None:
    result = run_market_research_workflow(
        price_data=make_price_data(),
        static_weights=make_static_weights(),
        regime_weight_policy=make_regime_policy(),
        config=MarketResearchWorkflowConfig(
            n_regimes=3,
            feature_window=5,
            random_state=7,
        ),
    )

    assert result.dynamic_applied_weight_frame.index.min() > (
        result.dynamic_target_weight_frame.index.min()
    )


def test_market_research_workflow_rejects_missing_price_column() -> None:
    price_data = make_price_data().drop(columns=["adjusted_close"])

    with pytest.raises(MarketResearchWorkflowError, match="price column"):
        run_market_research_workflow(
            price_data=price_data,
            static_weights=make_static_weights(),
            regime_weight_policy=make_regime_policy(),
        )


def test_market_research_workflow_rejects_duplicate_prices() -> None:
    price_data = pd.concat(
        [
            make_price_data(),
            make_price_data().iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(MarketResearchWorkflowError, match="duplicate"):
        run_market_research_workflow(
            price_data=price_data,
            static_weights=make_static_weights(),
            regime_weight_policy=make_regime_policy(),
        )


def test_market_research_workflow_rejects_bad_static_weights() -> None:
    static_weights = {
        "SPY": 0.50,
        "TLT": 0.30,
        "GLD": 0.10,
    }

    with pytest.raises(MarketResearchWorkflowError, match="sum to 1.0"):
        run_market_research_workflow(
            price_data=make_price_data(),
            static_weights=static_weights,
            regime_weight_policy=make_regime_policy(),
        )


def test_market_research_workflow_rejects_missing_regime_policy() -> None:
    regime_policy = {
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
    }

    with pytest.raises(MarketResearchWorkflowError, match="Regime policy"):
        run_market_research_workflow(
            price_data=make_price_data(),
            static_weights=make_static_weights(),
            regime_weight_policy=regime_policy,
        )


def test_market_research_workflow_rejects_too_few_regimes() -> None:
    with pytest.raises(MarketResearchWorkflowError, match="At least two"):
        run_market_research_workflow(
            price_data=make_price_data(),
            static_weights=make_static_weights(),
            regime_weight_policy=make_regime_policy(),
            config=MarketResearchWorkflowConfig(n_regimes=1),
        )


def test_market_research_workflow_rejects_too_large_feature_window() -> None:
    with pytest.raises(MarketResearchWorkflowError, match="Feature matrix is empty"):
        run_market_research_workflow(
            price_data=make_price_data(),
            static_weights=make_static_weights(),
            regime_weight_policy=make_regime_policy(),
            config=MarketResearchWorkflowConfig(
                n_regimes=3,
                feature_window=100,
            ),
        )
