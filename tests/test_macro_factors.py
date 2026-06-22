from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from regime_risk_engine.research.macro_factors import (
    MacroFactorInputError,
    MacroFactorSpec,
    build_macro_factor_matrix,
    infer_macro_factor_specs,
    read_macro_factor_csv,
    write_macro_factor_matrix_csv,
)


def test_build_macro_factor_matrix_transforms_level_and_difference() -> None:
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "policy_rate": [1.0, 1.25, 1.50],
            "credit_spread": [2.0, 2.4, 2.1],
        }
    )

    result = build_macro_factor_matrix(
        data,
        specs=[
            MacroFactorSpec(
                name="policy_rate_level",
                source_column="policy_rate",
                transformation="level",
            ),
            MacroFactorSpec(
                name="credit_spread_change",
                source_column="credit_spread",
                transformation="difference",
            ),
        ],
    )

    assert list(result.columns) == [
        "date",
        "policy_rate_level",
        "credit_spread_change",
    ]
    assert result["policy_rate_level"].tolist() == [1.0, 1.25, 1.5]
    assert np.isnan(result.loc[0, "credit_spread_change"])
    assert result.loc[1, "credit_spread_change"] == pytest.approx(0.4)


def test_build_macro_factor_matrix_transforms_percent_change() -> None:
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "cpi": [100.0, 101.0, 103.02],
        }
    )

    result = build_macro_factor_matrix(
        data,
        specs=[
            MacroFactorSpec(
                name="inflation_proxy",
                source_column="cpi",
                transformation="percent_change",
            )
        ],
    )

    assert np.isnan(result.loc[0, "inflation_proxy"])
    assert result.loc[1, "inflation_proxy"] == pytest.approx(0.01)
    assert result.loc[2, "inflation_proxy"] == pytest.approx(0.02)


def test_build_macro_factor_matrix_transforms_log_difference() -> None:
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "commodity_index": [100.0, 105.0, 110.0],
        }
    )

    result = build_macro_factor_matrix(
        data,
        specs=[
            MacroFactorSpec(
                name="commodity_log_change",
                source_column="commodity_index",
                transformation="log_difference",
            )
        ],
    )

    expected = np.log(105.0) - np.log(100.0)

    assert np.isnan(result.loc[0, "commodity_log_change"])
    assert result.loc[1, "commodity_log_change"] == pytest.approx(expected)


def test_build_macro_factor_matrix_supports_lagged_factors() -> None:
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "yield_curve": [0.5, 0.6, 0.8],
        }
    )

    result = build_macro_factor_matrix(
        data,
        specs=[
            MacroFactorSpec(
                name="lagged_yield_curve",
                source_column="yield_curve",
                lag_periods=1,
            )
        ],
    )

    assert np.isnan(result.loc[0, "lagged_yield_curve"])
    assert result.loc[1, "lagged_yield_curve"] == pytest.approx(0.5)
    assert result.loc[2, "lagged_yield_curve"] == pytest.approx(0.6)


def test_build_macro_factor_matrix_supports_standardization() -> None:
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "dollar_index": [100.0, 102.0, 104.0],
        }
    )

    result = build_macro_factor_matrix(
        data,
        specs=[
            MacroFactorSpec(
                name="dollar_z_score",
                source_column="dollar_index",
                standardize=True,
            )
        ],
    )

    assert result["dollar_z_score"].mean() == pytest.approx(0.0)
    assert result["dollar_z_score"].std() == pytest.approx(1.0)


def test_build_macro_factor_matrix_rejects_missing_date_column() -> None:
    data = pd.DataFrame({"policy_rate": [1.0, 1.25]})

    with pytest.raises(MacroFactorInputError, match="missing column"):
        build_macro_factor_matrix(
            data,
            specs=[
                MacroFactorSpec(
                    name="policy_rate",
                    source_column="policy_rate",
                )
            ],
        )


def test_build_macro_factor_matrix_rejects_missing_source_column() -> None:
    data = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"]})

    with pytest.raises(MacroFactorInputError, match="missing column"):
        build_macro_factor_matrix(
            data,
            specs=[
                MacroFactorSpec(
                    name="policy_rate",
                    source_column="policy_rate",
                )
            ],
        )


def test_build_macro_factor_matrix_rejects_negative_lag() -> None:
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "policy_rate": [1.0, 1.25],
        }
    )

    with pytest.raises(MacroFactorInputError, match="lag_periods"):
        build_macro_factor_matrix(
            data,
            specs=[
                MacroFactorSpec(
                    name="policy_rate",
                    source_column="policy_rate",
                    lag_periods=-1,
                )
            ],
        )


def test_log_difference_rejects_non_positive_values() -> None:
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "commodity_index": [100.0, 0.0],
        }
    )

    with pytest.raises(MacroFactorInputError, match="positive values"):
        build_macro_factor_matrix(
            data,
            specs=[
                MacroFactorSpec(
                    name="commodity_log_change",
                    source_column="commodity_index",
                    transformation="log_difference",
                )
            ],
        )


def test_infer_macro_factor_specs_uses_numeric_columns() -> None:
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "policy_rate": [1.0, 1.25],
            "label": ["low", "higher"],
        }
    )

    specs = infer_macro_factor_specs(data)

    assert specs == [
        MacroFactorSpec(
            name="policy_rate",
            source_column="policy_rate",
            transformation="level",
        )
    ]


def test_read_macro_factor_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "macro.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "policy_rate": [1.0, 1.25],
        }
    ).to_csv(input_path, index=False)

    result = read_macro_factor_csv(
        input_path,
        specs=[
            MacroFactorSpec(
                name="policy_rate",
                source_column="policy_rate",
            )
        ],
    )

    assert result["policy_rate"].tolist() == [1.0, 1.25]


def test_write_macro_factor_matrix_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "macro_factors.csv"
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "policy_rate": [1.0, 1.25],
        }
    )

    result_path = write_macro_factor_matrix_csv(
        output_path,
        data=data,
        specs=[
            MacroFactorSpec(
                name="policy_rate",
                source_column="policy_rate",
            )
        ],
    )

    assert result_path == output_path
    assert output_path.exists()

    result = pd.read_csv(output_path)

    assert result["policy_rate"].tolist() == [1.0, 1.25]
