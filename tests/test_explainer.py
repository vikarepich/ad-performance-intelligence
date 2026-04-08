"""
Tests for the Explainer (src/ml/explainer.py).

What's happening here:
- We test SHAP computation, feature importance, and anomaly explanations
- We train a small model inside the fixture (not loading from disk)
  so tests don't depend on saved .pkl files
- We verify SHAP output shapes, types, and structure

Key testing challenge with SHAP:
- SHAP values depend on the model and data, so we can't check
  exact numbers — we check shapes, types, and structural properties
- We use a tiny dataset (30 rows) so tests run fast
"""

import pytest
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.ml.explainer import (
    load_model,
    compute_shap_values,
    get_feature_importance,
    get_top_features,
    explain_anomaly,
    save_shap_values,
    load_shap_values,
    ANOMALY_FEATURES,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_data():
    """
    Creates fake scaled feature data for SHAP testing.

    30 rows is enough for SHAP to work.
    Values are already "scaled" (centered around 0)
    to mimic StandardScaler output.
    """
    np.random.seed(42)
    n = 30

    data = {col: np.random.randn(n) for col in ANOMALY_FEATURES}
    return pd.DataFrame(data)


@pytest.fixture
def sample_target():
    """Creates a fake binary target: 9 anomalies, 21 normal."""
    return pd.Series([1] * 9 + [0] * 21)


@pytest.fixture
def trained_model(sample_data, sample_target):
    """
    Trains a small Logistic Regression for SHAP testing.

    We train a real model here (not load from disk) so tests
    don't depend on models/anomaly_model.pkl existing.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(sample_data, sample_target)
    return model


@pytest.fixture
def sample_shap_values(trained_model, sample_data):
    """Computes SHAP values from the trained model."""
    return compute_shap_values(trained_model, sample_data, model_name="test_model")


# ============================================================
# TESTS FOR load_model()
# ============================================================

class TestLoadModel:
    """Tests for load_model() — loading .pkl files."""

    def test_load_model_returns_three_items(self, trained_model, tmp_path):
        """Check that load_model returns model, scaler, and name."""
        pkl_path = tmp_path / "test_model.pkl"
        scaler = StandardScaler()
        scaler.fit(np.random.randn(10, 5))

        with open(pkl_path, "wb") as f:
            pickle.dump({
                "model": trained_model,
                "scaler": scaler,
                "name": "test_logistic",
            }, f)

        model, loaded_scaler, name = load_model(pkl_path)

        assert model is not None, "Model should not be None"
        assert loaded_scaler is not None, "Scaler should not be None"
        assert name == "test_logistic", f"Expected name 'test_logistic', got '{name}'"


# ============================================================
# TESTS FOR compute_shap_values()
# ============================================================

class TestComputeShapValues:
    """
    Tests for compute_shap_values().

    We check structural properties of SHAP output:
    - Correct shape (n_samples x n_features)
    - Returns numpy array
    - No NaN values
    """

    def test_shap_shape_matches_input(self, sample_shap_values, sample_data):
        """
        Check that SHAP output has the same shape as input data.

        SHAP produces one value per feature per sample:
        input shape (30, 10) -> SHAP shape (30, 10)
        """
        assert sample_shap_values.shape == sample_data.shape, \
            f"SHAP shape {sample_shap_values.shape} doesn't match data shape {sample_data.shape}"

    def test_shap_returns_numpy_array(self, sample_shap_values):
        """Check that SHAP values are a numpy array."""
        assert isinstance(sample_shap_values, np.ndarray), \
            f"SHAP should return numpy array, got {type(sample_shap_values)}"

    def test_shap_no_nan(self, sample_shap_values):
        """Check that SHAP values contain no NaN."""
        assert not np.isnan(sample_shap_values).any(), \
            "SHAP values contain NaN"

    def test_shap_no_inf(self, sample_shap_values):
        """Check that SHAP values contain no infinity."""
        assert not np.isinf(sample_shap_values).any(), \
            "SHAP values contain infinity"


# ============================================================
# TESTS FOR get_feature_importance()
# ============================================================

class TestGetFeatureImportance:
    """
    Tests for get_feature_importance().

    Global importance = mean(|SHAP|) per feature.
    """

    def test_returns_dataframe(self, sample_shap_values):
        """Check that result is a DataFrame."""
        result = get_feature_importance(sample_shap_values, ANOMALY_FEATURES)
        assert isinstance(result, pd.DataFrame), "Should return a DataFrame"

    def test_has_correct_columns(self, sample_shap_values):
        """Check that result has 'feature' and 'importance' columns."""
        result = get_feature_importance(sample_shap_values, ANOMALY_FEATURES)

        assert "feature" in result.columns, "Missing 'feature' column"
        assert "importance" in result.columns, "Missing 'importance' column"

    def test_all_features_present(self, sample_shap_values):
        """Check that all features appear in the importance table."""
        result = get_feature_importance(sample_shap_values, ANOMALY_FEATURES)

        result_features = set(result["feature"].tolist())
        expected_features = set(ANOMALY_FEATURES)
        assert result_features == expected_features, \
            f"Missing features: {expected_features - result_features}"

    def test_importance_non_negative(self, sample_shap_values):
        """
        Check that all importance values are >= 0.

        Importance = mean(|SHAP|), so it can never be negative.
        """
        result = get_feature_importance(sample_shap_values, ANOMALY_FEATURES)

        assert (result["importance"] >= 0).all(), \
            "Importance values should be non-negative"

    def test_sorted_descending(self, sample_shap_values):
        """Check that features are sorted by importance (highest first)."""
        result = get_feature_importance(sample_shap_values, ANOMALY_FEATURES)

        importance_values = result["importance"].tolist()
        assert importance_values == sorted(importance_values, reverse=True), \
            "Importance should be sorted descending"


# ============================================================
# TESTS FOR get_top_features()
# ============================================================

class TestGetTopFeatures:
    """Tests for get_top_features() — local importance for one row."""

    def test_returns_correct_count(self, sample_shap_values):
        """Check that we get exactly top_n features."""
        row_shap = sample_shap_values[0]

        result = get_top_features(row_shap, ANOMALY_FEATURES, top_n=3)
        assert len(result) == 3, f"Expected 3 top features, got {len(result)}"

        result = get_top_features(row_shap, ANOMALY_FEATURES, top_n=5)
        assert len(result) == 5, f"Expected 5 top features, got {len(result)}"

    def test_returns_correct_structure(self, sample_shap_values):
        """Check that each item has feature, shap_value, and rank."""
        row_shap = sample_shap_values[0]
        result = get_top_features(row_shap, ANOMALY_FEATURES, top_n=3)

        for item in result:
            assert "feature" in item, "Missing 'feature' key"
            assert "shap_value" in item, "Missing 'shap_value' key"
            assert "rank" in item, "Missing 'rank' key"

    def test_ranks_are_sequential(self, sample_shap_values):
        """Check that ranks are 1, 2, 3, ... in order."""
        row_shap = sample_shap_values[0]
        result = get_top_features(row_shap, ANOMALY_FEATURES, top_n=3)

        ranks = [item["rank"] for item in result]
        assert ranks == [1, 2, 3], f"Ranks should be [1, 2, 3], got {ranks}"

    def test_sorted_by_absolute_shap(self, sample_shap_values):
        """
        Check that features are sorted by |SHAP| (highest impact first).

        The first feature should have the largest absolute SHAP value.
        """
        row_shap = sample_shap_values[0]
        result = get_top_features(row_shap, ANOMALY_FEATURES, top_n=3)

        abs_values = [abs(item["shap_value"]) for item in result]
        assert abs_values == sorted(abs_values, reverse=True), \
            "Top features should be sorted by absolute SHAP value"


# ============================================================
# TESTS FOR explain_anomaly()
# ============================================================

class TestExplainAnomaly:
    """Tests for explain_anomaly() — human-readable explanations."""

    def test_returns_correct_structure(self, sample_shap_values, sample_data):
        """Check that result has row_index, top_factors, and explanation."""
        result = explain_anomaly(
            row_index=0,
            shap_values=sample_shap_values,
            X=sample_data,
            feature_names=ANOMALY_FEATURES,
        )

        assert "row_index" in result, "Missing 'row_index'"
        assert "top_factors" in result, "Missing 'top_factors'"
        assert "explanation" in result, "Missing 'explanation'"

    def test_explanation_is_string(self, sample_shap_values, sample_data):
        """Check that explanation is a non-empty string."""
        result = explain_anomaly(
            row_index=0,
            shap_values=sample_shap_values,
            X=sample_data,
            feature_names=ANOMALY_FEATURES,
        )

        assert isinstance(result["explanation"], str), "Explanation should be a string"
        assert len(result["explanation"]) > 0, "Explanation should not be empty"

    def test_explanation_contains_direction(self, sample_shap_values, sample_data):
        """
        Check that explanation contains direction words.

        Every explanation should say whether a feature pushes
        TOWARD or AWAY from anomaly.
        """
        result = explain_anomaly(
            row_index=0,
            shap_values=sample_shap_values,
            X=sample_data,
            feature_names=ANOMALY_FEATURES,
        )

        explanation = result["explanation"]
        has_direction = "TOWARD" in explanation or "AWAY" in explanation
        assert has_direction, "Explanation should contain 'TOWARD' or 'AWAY'"

    def test_row_index_matches(self, sample_shap_values, sample_data):
        """Check that the returned row_index matches the input."""
        result = explain_anomaly(
            row_index=5,
            shap_values=sample_shap_values,
            X=sample_data,
            feature_names=ANOMALY_FEATURES,
        )

        assert result["row_index"] == 5, \
            f"Expected row_index=5, got {result['row_index']}"


# ============================================================
# TESTS FOR save/load SHAP values
# ============================================================

class TestSaveLoadShap:
    """Tests for save_shap_values() and load_shap_values()."""

    def test_save_creates_file(self, sample_shap_values, tmp_path):
        """Check that save creates a .pkl file."""
        output_path = tmp_path / "test_shap.pkl"

        save_shap_values(sample_shap_values, ANOMALY_FEATURES, output_path)

        assert output_path.exists(), "SHAP .pkl file was not created"

    def test_save_load_roundtrip(self, sample_shap_values, tmp_path):
        """
        Check that saving and loading produces the same data.

        This is a "roundtrip" test: save -> load -> compare.
        If the data is the same after the roundtrip, both functions work.
        """
        output_path = tmp_path / "test_shap.pkl"

        save_shap_values(sample_shap_values, ANOMALY_FEATURES, output_path)

        loaded_values, loaded_names = load_shap_values(output_path)

        np.testing.assert_array_almost_equal(
            loaded_values, sample_shap_values,
            err_msg="Loaded SHAP values don't match saved values"
        )
        assert loaded_names == ANOMALY_FEATURES, \
            "Loaded feature names don't match saved names"