import pandas as pd
import pytest

from regime_risk_engine.regimes.base import RegimeModelError
from regime_risk_engine.regimes.gmm import (
    GaussianMixtureRegimeConfig,
    GaussianMixtureRegimeModel,
)


def make_feature_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [
                -3.0,
                -2.8,
                -3.2,
                -2.9,
                0.0,
                0.1,
                -0.1,
                0.2,
                3.0,
                2.8,
                3.2,
                2.9,
            ],
            "feature_b": [
                -3.0,
                -2.9,
                -3.1,
                -2.8,
                0.0,
                0.2,
                -0.2,
                0.1,
                3.0,
                2.9,
                3.1,
                2.8,
            ],
        },
        index=pd.date_range("2020-01-01", periods=12, freq="D"),
    )


def test_gmm_regime_model_fit_sets_model_state() -> None:
    feature_matrix = make_feature_matrix()
    model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(
            n_regimes=3,
            random_state=42,
        )
    )

    fitted_model = model.fit(feature_matrix)

    assert fitted_model is model
    assert model.is_fitted is True
    assert model.feature_columns == ["feature_a", "feature_b"]


def test_gmm_regime_model_predict_returns_labels() -> None:
    feature_matrix = make_feature_matrix()
    model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(n_regimes=3, random_state=42)
    )

    model.fit(feature_matrix)
    labels = model.predict(feature_matrix)

    assert isinstance(labels, pd.Series)
    assert labels.name == "regime"
    assert labels.index.equals(feature_matrix.index)
    assert len(labels) == len(feature_matrix)
    assert set(labels.unique()) == {0, 1, 2}


def test_gmm_regime_model_predict_proba_returns_probabilities() -> None:
    feature_matrix = make_feature_matrix()
    model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(n_regimes=3, random_state=42)
    )

    model.fit(feature_matrix)
    probabilities = model.predict_proba(feature_matrix)

    assert isinstance(probabilities, pd.DataFrame)
    assert probabilities.index.equals(feature_matrix.index)
    assert list(probabilities.columns) == [
        "regime_0_probability",
        "regime_1_probability",
        "regime_2_probability",
    ]
    assert probabilities.shape == (12, 3)
    assert probabilities.sum(axis=1).to_list() == pytest.approx([1.0] * 12)


def test_gmm_regime_model_fit_predict_returns_labels() -> None:
    feature_matrix = make_feature_matrix()
    model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(n_regimes=3, random_state=42)
    )

    labels = model.fit_predict(feature_matrix)

    assert isinstance(labels, pd.Series)
    assert len(labels) == len(feature_matrix)
    assert set(labels.unique()) == {0, 1, 2}


def test_gmm_regime_model_is_deterministic_with_fixed_seed() -> None:
    feature_matrix = make_feature_matrix()

    first_model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(n_regimes=3, random_state=42)
    )
    second_model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(n_regimes=3, random_state=42)
    )

    first_labels = first_model.fit_predict(feature_matrix)
    second_labels = second_model.fit_predict(feature_matrix)

    pd.testing.assert_series_equal(first_labels, second_labels)

    first_probabilities = first_model.predict_proba(feature_matrix)
    second_probabilities = second_model.predict_proba(feature_matrix)

    pd.testing.assert_frame_equal(first_probabilities, second_probabilities)


def test_gmm_regime_model_fit_predict_result() -> None:
    feature_matrix = make_feature_matrix()
    model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(n_regimes=3, random_state=42)
    )

    result = model.fit_predict_result(feature_matrix)

    assert result.model_name == "gaussian_mixture"
    assert result.metadata["n_regimes"] == 3
    assert result.metadata["covariance_type"] == "full"
    assert result.metadata["n_features"] == 2
    assert result.metadata["n_observations"] == 12
    assert len(result.labels) == len(feature_matrix)


def test_gmm_regime_model_rejects_invalid_n_regimes() -> None:
    with pytest.raises(RegimeModelError, match="greater than 1"):
        GaussianMixtureRegimeModel(GaussianMixtureRegimeConfig(n_regimes=1))


def test_gmm_regime_model_rejects_invalid_covariance_type() -> None:
    with pytest.raises(RegimeModelError, match="Covariance type"):
        GaussianMixtureRegimeModel(
            GaussianMixtureRegimeConfig(covariance_type="invalid")
        )


def test_gmm_regime_model_rejects_prediction_before_fit() -> None:
    feature_matrix = make_feature_matrix()
    model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(n_regimes=3, random_state=42)
    )

    with pytest.raises(RegimeModelError, match="has not been fit"):
        model.predict(feature_matrix)


def test_gmm_regime_model_rejects_proba_before_fit() -> None:
    feature_matrix = make_feature_matrix()
    model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(n_regimes=3, random_state=42)
    )

    with pytest.raises(RegimeModelError, match="has not been fit"):
        model.predict_proba(feature_matrix)


def test_gmm_regime_model_rejects_missing_values() -> None:
    feature_matrix = make_feature_matrix()
    feature_matrix.iloc[0, 0] = None

    model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(n_regimes=3, random_state=42)
    )

    with pytest.raises(RegimeModelError, match="missing values"):
        model.fit(feature_matrix)


def test_gmm_regime_model_rejects_non_datetime_index() -> None:
    feature_matrix = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [1.0, 2.0, 3.0],
        },
        index=[0, 1, 2],
    )
    model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(n_regimes=2, random_state=42)
    )

    with pytest.raises(RegimeModelError, match="DatetimeIndex"):
        model.fit(feature_matrix)


def test_gmm_regime_model_rejects_changed_feature_columns() -> None:
    feature_matrix = make_feature_matrix()
    model = GaussianMixtureRegimeModel(
        GaussianMixtureRegimeConfig(n_regimes=3, random_state=42)
    )
    model.fit(feature_matrix)

    changed_matrix = feature_matrix.rename(columns={"feature_a": "different_feature"})

    with pytest.raises(RegimeModelError, match="Feature columns must match"):
        model.predict(changed_matrix)
