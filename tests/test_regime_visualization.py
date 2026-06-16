import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from regime_risk_engine.regimes.visualization import (
    RegimeVisualizationError,
    plot_cumulative_returns_with_regimes,
    plot_regime_probabilities,
    plot_regime_timeline,
)


def make_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")

    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["SPY"] * 5 + ["TLT"] * 5,
            "return": [
                0.01,
                0.02,
                -0.01,
                0.00,
                0.01,
                0.00,
                -0.01,
                0.02,
                0.01,
                0.00,
            ],
        }
    )


def make_regime_labels() -> pd.Series:
    return pd.Series(
        [0, 0, 1, 1, 0],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
        name="regime",
    )


def make_probabilities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regime_0_probability": [0.9, 0.8, 0.2, 0.1, 0.7],
            "regime_1_probability": [0.1, 0.2, 0.8, 0.9, 0.3],
        },
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )


def test_plot_regime_timeline_returns_figure() -> None:
    labels = make_regime_labels()

    figure = plot_regime_timeline(labels)

    assert isinstance(figure, Figure)

    plt.close(figure)


def test_plot_regime_timeline_with_label_map_returns_figure() -> None:
    labels = make_regime_labels()
    label_map = {
        0: "bull_low_volatility",
        1: "high_volatility_stress",
    }

    figure = plot_regime_timeline(labels, label_map=label_map)

    assert isinstance(figure, Figure)

    plt.close(figure)


def test_plot_regime_probabilities_returns_figure() -> None:
    probabilities = make_probabilities()

    figure = plot_regime_probabilities(probabilities)

    assert isinstance(figure, Figure)

    plt.close(figure)


def test_plot_cumulative_returns_with_regimes_returns_figure() -> None:
    returns = make_returns()
    labels = make_regime_labels()

    figure = plot_cumulative_returns_with_regimes(returns, labels)

    assert isinstance(figure, Figure)

    plt.close(figure)


def test_plot_regime_timeline_rejects_non_datetime_index() -> None:
    labels = pd.Series([0, 1], index=[0, 1], name="regime")

    with pytest.raises(RegimeVisualizationError, match="DatetimeIndex"):
        plot_regime_timeline(labels)


def test_plot_regime_probabilities_rejects_invalid_row_sums() -> None:
    probabilities = pd.DataFrame(
        {
            "regime_0_probability": [0.7, 0.8],
            "regime_1_probability": [0.1, 0.1],
        },
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    with pytest.raises(RegimeVisualizationError, match="sum to 1.0"):
        plot_regime_probabilities(probabilities)


def test_plot_cumulative_returns_rejects_missing_return_columns() -> None:
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"]),
            "ticker": ["SPY"],
        }
    )
    labels = make_regime_labels()

    with pytest.raises(RegimeVisualizationError, match="Missing required return"):
        plot_cumulative_returns_with_regimes(returns, labels)


def test_plot_cumulative_returns_rejects_no_overlapping_dates() -> None:
    returns = make_returns()
    labels = pd.Series(
        [0, 1],
        index=pd.to_datetime(["2030-01-01", "2030-01-02"]),
        name="regime",
    )

    with pytest.raises(RegimeVisualizationError, match="no overlapping dates"):
        plot_cumulative_returns_with_regimes(returns, labels)
