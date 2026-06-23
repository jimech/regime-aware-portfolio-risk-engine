from pathlib import Path

import pandas as pd
import pytest

from regime_risk_engine.research.advanced_cli_export import (
    AdvancedResearchCliExportError,
    export_advanced_research_from_files,
    format_advanced_export_result,
)


def make_price_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rows = []

    prices = {
        "SPY": 100.0,
        "TLT": 100.0,
        "GLD": 100.0,
    }

    for index, date in enumerate(dates):
        if index < 20:
            returns = {"SPY": 0.004, "TLT": 0.0005, "GLD": 0.0002}
        elif index < 40:
            returns = {"SPY": -0.006, "TLT": 0.003, "GLD": 0.001}
        else:
            returns = {"SPY": 0.0005, "TLT": -0.001, "GLD": 0.004}

        for ticker, ticker_return in returns.items():
            prices[ticker] *= 1.0 + ticker_return
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_close": prices[ticker],
                }
            )

    return pd.DataFrame(rows)


def write_static_weights(path: Path) -> None:
    pd.DataFrame(
        {
            "ticker": ["SPY", "TLT", "GLD"],
            "weight": [0.60, 0.30, 0.10],
        }
    ).to_csv(path, index=False)


def write_regime_policy(path: Path) -> None:
    pd.DataFrame(
        [
            {"regime": 0, "ticker": "SPY", "weight": 0.70},
            {"regime": 0, "ticker": "TLT", "weight": 0.20},
            {"regime": 0, "ticker": "GLD", "weight": 0.10},
            {"regime": 1, "ticker": "SPY", "weight": 0.30},
            {"regime": 1, "ticker": "TLT", "weight": 0.50},
            {"regime": 1, "ticker": "GLD", "weight": 0.20},
            {"regime": 2, "ticker": "SPY", "weight": 0.40},
            {"regime": 2, "ticker": "TLT", "weight": 0.20},
            {"regime": 2, "ticker": "GLD", "weight": 0.40},
        ]
    ).to_csv(path, index=False)


def write_stress_periods(path: Path) -> None:
    pd.DataFrame(
        {
            "name": ["Full sample stress review"],
            "start_date": ["2020-01-02"],
            "end_date": ["2020-02-29"],
        }
    ).to_csv(path, index=False)


def write_factor_returns(path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=60, freq="D")

    pd.DataFrame(
        {
            "date": dates,
            "equity": [0.004 if index < 20 else -0.006 for index in range(60)],
            "defensive": [0.0005 if index < 40 else -0.001 for index in range(60)],
            "real_asset": [0.0002 if index < 40 else 0.004 for index in range(60)],
        }
    ).to_csv(path, index=False)


def test_export_advanced_research_from_files(tmp_path: Path) -> None:
    price_path = tmp_path / "prices.csv"
    static_weights_path = tmp_path / "static_weights.csv"
    regime_policy_path = tmp_path / "regime_policy.csv"
    stress_periods_path = tmp_path / "stress_periods.csv"
    factor_returns_path = tmp_path / "factor_returns.csv"

    make_price_data().to_csv(price_path, index=False)
    write_static_weights(static_weights_path)
    write_regime_policy(regime_policy_path)
    write_stress_periods(stress_periods_path)
    write_factor_returns(factor_returns_path)

    result = export_advanced_research_from_files(
        price_data_path=price_path,
        static_weights_path=static_weights_path,
        regime_policy_path=regime_policy_path,
        stress_periods_path=stress_periods_path,
        factor_returns_path=factor_returns_path,
        output_dir=tmp_path / "advanced_export",
        n_regimes=3,
        feature_window=5,
        scenario_horizon=5,
        scenario_simulations=25,
        random_state=7,
        analyst="Jimena Chinchilla",
    )

    assert result.memo_path.exists()
    assert "regime_intelligence_profile" in result.exported_table_paths
    assert "stress_test_summary" in result.exported_table_paths
    assert "factor_exposure" in result.exported_table_paths
    assert "rolling_factor_exposures" in result.exported_table_paths
    assert "rolling_factor_exposure_summary" in result.exported_table_paths
    assert "scenario_terminal_summary" in result.exported_table_paths

    cli_output = format_advanced_export_result(result)

    assert "Advanced research package exported successfully" in cli_output
    assert "advanced_research_memo.md" in cli_output


def test_export_advanced_research_from_files_works_without_optional_files(
    tmp_path: Path,
) -> None:
    price_path = tmp_path / "prices.csv"
    static_weights_path = tmp_path / "static_weights.csv"

    make_price_data().to_csv(price_path, index=False)
    write_static_weights(static_weights_path)

    result = export_advanced_research_from_files(
        price_data_path=price_path,
        static_weights_path=static_weights_path,
        output_dir=tmp_path / "advanced_export",
        n_regimes=3,
        feature_window=5,
        scenario_horizon=5,
        scenario_simulations=25,
        random_state=7,
    )

    assert result.memo_path.exists()
    assert "regime_intelligence_profile" in result.exported_table_paths
    assert "scenario_terminal_summary" in result.exported_table_paths
    assert "rolling_factor_exposures" not in result.exported_table_paths
    assert "rolling_factor_exposure_summary" not in result.exported_table_paths


def test_export_advanced_research_rejects_missing_price_file(
    tmp_path: Path,
) -> None:
    static_weights_path = tmp_path / "static_weights.csv"
    write_static_weights(static_weights_path)

    with pytest.raises(AdvancedResearchCliExportError, match="Missing price data"):
        export_advanced_research_from_files(
            price_data_path=tmp_path / "missing.csv",
            static_weights_path=static_weights_path,
            output_dir=tmp_path / "advanced_export",
        )


def test_export_advanced_research_rejects_bad_static_weight_sum(
    tmp_path: Path,
) -> None:
    price_path = tmp_path / "prices.csv"
    static_weights_path = tmp_path / "static_weights.csv"

    make_price_data().to_csv(price_path, index=False)

    pd.DataFrame(
        {
            "ticker": ["SPY", "TLT", "GLD"],
            "weight": [0.50, 0.30, 0.10],
        }
    ).to_csv(static_weights_path, index=False)

    with pytest.raises(AdvancedResearchCliExportError, match="sum to 1.0"):
        export_advanced_research_from_files(
            price_data_path=price_path,
            static_weights_path=static_weights_path,
            output_dir=tmp_path / "advanced_export",
        )


def test_export_advanced_research_rejects_policy_with_missing_ticker(
    tmp_path: Path,
) -> None:
    price_path = tmp_path / "prices.csv"
    static_weights_path = tmp_path / "static_weights.csv"
    regime_policy_path = tmp_path / "regime_policy.csv"

    make_price_data().to_csv(price_path, index=False)
    write_static_weights(static_weights_path)

    pd.DataFrame(
        [
            {"regime": 0, "ticker": "SPY", "weight": 0.70},
            {"regime": 0, "ticker": "TLT", "weight": 0.30},
        ]
    ).to_csv(regime_policy_path, index=False)

    with pytest.raises(AdvancedResearchCliExportError, match="tickers"):
        export_advanced_research_from_files(
            price_data_path=price_path,
            static_weights_path=static_weights_path,
            regime_policy_path=regime_policy_path,
            output_dir=tmp_path / "advanced_export",
        )


def test_export_advanced_research_rejects_invalid_rolling_factor_window(
    tmp_path: Path,
) -> None:
    price_path = tmp_path / "prices.csv"
    static_weights_path = tmp_path / "static_weights.csv"

    make_price_data().to_csv(price_path, index=False)
    write_static_weights(static_weights_path)

    with pytest.raises(AdvancedResearchCliExportError, match="Rolling factor window"):
        export_advanced_research_from_files(
            price_data_path=price_path,
            static_weights_path=static_weights_path,
            output_dir=tmp_path / "advanced_export",
            rolling_factor_window=0,
        )
