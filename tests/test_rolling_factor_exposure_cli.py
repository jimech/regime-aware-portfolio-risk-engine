import json
from pathlib import Path

import pandas as pd

from regime_risk_engine.cli import main


def test_cli_export_rolling_factor_exposure(tmp_path: Path, capsys) -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    equity = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.03, 0.01]
    strategy = [0.001 + 2.0 * value for value in equity]

    strategy_path = tmp_path / "strategy_returns.csv"
    factor_path = tmp_path / "factor_returns.csv"
    output_dir = tmp_path / "rolling_factor_exposure"

    pd.DataFrame({"date": dates, "return": strategy}).to_csv(
        strategy_path,
        index=False,
    )
    pd.DataFrame({"date": dates, "equity": equity}).to_csv(
        factor_path,
        index=False,
    )

    exit_code = main(
        [
            "export-rolling-factor-exposure",
            "--strategy-returns",
            str(strategy_path),
            "--factor-returns",
            str(factor_path),
            "--output-dir",
            str(output_dir),
            "--window",
            "5",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Rolling factor exposure exported successfully" in captured.out
    assert (output_dir / "rolling_factor_exposures.csv").exists()
    assert (output_dir / "rolling_factor_exposure_summary.csv").exists()

    exposures = pd.read_csv(output_dir / "rolling_factor_exposures.csv")
    summary = pd.read_csv(output_dir / "rolling_factor_exposure_summary.csv")

    assert "equity_beta" in exposures.columns
    assert summary.loc[0, "factor"] == "equity"


def test_cli_export_rolling_factor_exposure_json(tmp_path: Path, capsys) -> None:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    equity = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.03, 0.01]
    strategy = [2.0 * value for value in equity]

    strategy_path = tmp_path / "strategy_returns.csv"
    factor_path = tmp_path / "factor_returns.csv"
    output_dir = tmp_path / "rolling_factor_exposure"

    pd.DataFrame({"date": dates, "return": strategy}).to_csv(
        strategy_path,
        index=False,
    )
    pd.DataFrame({"date": dates, "equity": equity}).to_csv(
        factor_path,
        index=False,
    )

    exit_code = main(
        [
            "export-rolling-factor-exposure",
            "--strategy-returns",
            str(strategy_path),
            "--factor-returns",
            str(factor_path),
            "--output-dir",
            str(output_dir),
            "--window",
            "5",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["factor_columns"] == ["equity"]
    assert payload["window"] == 5
    assert Path(payload["rolling_factor_exposures_path"]).exists()


def test_cli_export_rolling_factor_exposure_returns_error_for_bad_input(
    tmp_path: Path,
    capsys,
) -> None:
    factor_path = tmp_path / "factor_returns.csv"
    output_dir = tmp_path / "rolling_factor_exposure"

    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "equity": [0.01, 0.02, 0.03, 0.01, 0.02],
        }
    ).to_csv(factor_path, index=False)

    exit_code = main(
        [
            "export-rolling-factor-exposure",
            "--strategy-returns",
            str(tmp_path / "missing.csv"),
            "--factor-returns",
            str(factor_path),
            "--output-dir",
            str(output_dir),
            "--window",
            "3",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Rolling factor exposure export failed" in captured.out
