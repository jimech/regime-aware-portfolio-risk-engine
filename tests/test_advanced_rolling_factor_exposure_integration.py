from pathlib import Path

import pandas as pd

from regime_risk_engine.research.advanced_cli_export import (
    export_advanced_research_from_files,
)


def test_advanced_export_includes_rolling_factor_exposure_tables(
    tmp_path: Path,
) -> None:
    dates = pd.date_range("2020-01-01", periods=80, freq="D")
    rows = []
    prices = {"SPY": 100.0, "TLT": 100.0, "GLD": 100.0}

    for index, date in enumerate(dates):
        returns = {
            "SPY": 0.003 if index < 40 else -0.002,
            "TLT": 0.001 if index < 40 else 0.002,
            "GLD": 0.0005 if index < 40 else 0.003,
        }

        for ticker, ticker_return in returns.items():
            prices[ticker] *= 1.0 + ticker_return
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adjusted_close": prices[ticker],
                }
            )

    price_path = tmp_path / "prices.csv"
    static_weights_path = tmp_path / "static_weights.csv"
    factor_returns_path = tmp_path / "factor_returns.csv"
    output_dir = tmp_path / "advanced_export"

    pd.DataFrame(rows).to_csv(price_path, index=False)
    pd.DataFrame(
        {
            "ticker": ["SPY", "TLT", "GLD"],
            "weight": [0.60, 0.30, 0.10],
        }
    ).to_csv(static_weights_path, index=False)
    pd.DataFrame(
        {
            "date": dates,
            "equity": [0.003 if index < 40 else -0.002 for index in range(80)],
            "defensive": [0.001 if index < 40 else 0.002 for index in range(80)],
            "real_asset": [0.0005 if index < 40 else 0.003 for index in range(80)],
        }
    ).to_csv(factor_returns_path, index=False)

    result = export_advanced_research_from_files(
        price_data_path=price_path,
        static_weights_path=static_weights_path,
        factor_returns_path=factor_returns_path,
        output_dir=output_dir,
        n_regimes=3,
        feature_window=5,
        scenario_horizon=5,
        scenario_simulations=25,
        random_state=7,
    )

    assert "rolling_factor_exposures" in result.exported_table_paths
    assert "rolling_factor_exposure_summary" in result.exported_table_paths

    rolling_path = result.exported_table_paths["rolling_factor_exposures"]
    summary_path = result.exported_table_paths["rolling_factor_exposure_summary"]

    assert rolling_path.exists()
    assert summary_path.exists()

    rolling = pd.read_csv(rolling_path)
    summary = pd.read_csv(summary_path)
    memo = result.memo_path.read_text(encoding="utf-8")

    assert "equity_beta" in rolling.columns
    assert "factor" in summary.columns
    assert "Rolling Factor Exposure Analysis" in memo
