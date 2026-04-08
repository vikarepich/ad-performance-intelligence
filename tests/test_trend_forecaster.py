"""
Tests for the Trend Forecaster (src/ml/trend_forecaster.py).

What's happening here:
- We test all key functions: prepare_data, train_*, evaluate_model, save_best_model
- Same fixture approach as test_anomaly_detector.py
- Key difference: regression metrics (MAE, RMSE, R²) instead of classification

New concept — regression test gotchas:
- R² can be negative (model worse than mean), so we don't check R² >= 0
  on random fake data — only that it returns a float
- MAE and RMSE are always >= 0
- Predictions are floats, not class labels (0/1)
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.ml.trend_forecaster import (
    prepare_data,
    train_linear,
    train_xgboost,
    train_lightgbm,
    evaluate_model,
    save_best_model,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


# ============================================================
# FIXTURE — fake feature-engineered data
# ============================================================

@pytest.fixture
def sample_features_df():
    """
    Creates a fake DataFrame that mimics features.csv for regression.

    50 rows with random but realistic values.
    roas is the target — we generate it with some correlation
    to other features so models have something to learn.
    """
    np.random.seed(42)
    n = 50

    ctr = np.random.uniform(0.01, 0.10, n)
    cpc = np.random.uniform(0.5, 5.0, n)

    data = {
        "ctr": ctr,
        "cpc": cpc,
        "cpl": np.random.uniform(5.0, 50.0, n),
        "conversion_rate": np.random.uniform(0.01, 0.15, n),
        "ctr_wow": np.random.uniform(-0.5, 0.5, n),
        "spend_wow": np.random.uniform(-0.5, 0.5, n),
        "ctr_rolling3": np.random.uniform(0.02, 0.08, n),
        # roas has some correlation with ctr and inverse with cpc
        # so models can actually learn patterns (not pure noise)
        "roas": 2.0 + ctr * 30 - cpc * 0.3 + np.random.normal(0, 0.5, n),
    }

    return pd.DataFrame(data)


@pytest.fixture
def trained_data(sample_features_df):
    """
    Prepares train/test split from the fake data.

    Returns a dictionary with all pieces needed for testing.
    """
    X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


# ============================================================
# TESTS FOR prepare_data()
# ============================================================

class TestPrepareData:
    """Tests for prepare_data() — same checks as anomaly detector."""

    def test_returns_correct_shapes(self, sample_features_df):
        """Check that train/test sets have the right dimensions."""
        X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)

        assert len(X_train) + len(X_test) == len(sample_features_df), \
            "Train + test rows don't add up to original"

        assert X_train.shape[1] == len(FEATURE_COLUMNS), \
            f"Expected {len(FEATURE_COLUMNS)} features, got {X_train.shape[1]}"

    def test_target_is_continuous(self, sample_features_df):
        """
        Check that the target (roas) is continuous, not binary.

        This catches a common mistake: accidentally using
        is_anomaly as target instead of roas.
        """
        X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)

        assert y_train.nunique() > 2, \
            "Target should be continuous (many unique values), not binary"

    def test_scaler_is_fitted(self, sample_features_df):
        """Check that the scaler has learned mean and std."""
        X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)

        assert hasattr(scaler, "mean_"), "Scaler was not fitted (no mean_)"
        assert len(scaler.mean_) == len(FEATURE_COLUMNS), \
            "Scaler mean_ length doesn't match feature count"

    def test_no_nan_after_preparation(self, sample_features_df):
        """Check that there are no NaN values after preparation."""
        X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)

        assert X_train.isna().sum().sum() == 0, "X_train contains NaN"
        assert X_test.isna().sum().sum() == 0, "X_test contains NaN"

    def test_no_roas_in_features(self, sample_features_df):
        """
        Check that roas is NOT in the feature columns.

        This is a data leakage test — if roas is in the features,
        the model is cheating (it already knows the answer).
        """
        assert "roas" not in FEATURE_COLUMNS, \
            "roas should NOT be in FEATURE_COLUMNS (data leakage!)"
        assert "roas_wow" not in FEATURE_COLUMNS, \
            "roas_wow should NOT be in FEATURE_COLUMNS (data leakage!)"
        assert "roas_rolling3" not in FEATURE_COLUMNS, \
            "roas_rolling3 should NOT be in FEATURE_COLUMNS (data leakage!)"


# ============================================================
# TESTS FOR train functions
# ============================================================

class TestTrainModels:
    """
    Tests for each train_* function.

    For regression models we check:
    1. Returns the correct model type
    2. Can make predictions
    3. Predictions are floats (not class labels)
    """

    def test_train_linear(self, trained_data):
        """Check that Linear Regression trains and predicts."""
        model = train_linear(trained_data["X_train"], trained_data["y_train"])

        assert isinstance(model, LinearRegression), \
            "Should return a LinearRegression instance"

        predictions = model.predict(trained_data["X_test"])
        assert len(predictions) == len(trained_data["X_test"]), \
            "Prediction count doesn't match test set size"

        # Regression predictions should be floats, not integers
        assert predictions.dtype in [np.float64, np.float32], \
            f"Predictions should be float, got {predictions.dtype}"

    def test_train_xgboost(self, trained_data):
        """Check that XGBoost Regressor trains and predicts."""
        model = train_xgboost(trained_data["X_train"], trained_data["y_train"])

        assert isinstance(model, XGBRegressor), \
            "Should return an XGBRegressor instance"

        predictions = model.predict(trained_data["X_test"])
        assert len(predictions) == len(trained_data["X_test"]), \
            "Prediction count doesn't match test set size"

    def test_train_lightgbm(self, trained_data):
        """Check that LightGBM Regressor trains and predicts."""
        model = train_lightgbm(trained_data["X_train"], trained_data["y_train"])

        assert isinstance(model, LGBMRegressor), \
            "Should return an LGBMRegressor instance"

        predictions = model.predict(trained_data["X_test"])
        assert len(predictions) == len(trained_data["X_test"]), \
            "Prediction count doesn't match test set size"


# ============================================================
# TESTS FOR evaluate_model()
# ============================================================

class TestEvaluateModel:
    """
    Tests for evaluate_model().

    Regression metrics have different valid ranges than classification:
    - MAE >= 0 (lower is better)
    - RMSE >= 0 (lower is better)
    - R² can be any value (even negative), but typically -1 to 1
    """

    def test_returns_all_metrics(self, trained_data):
        """Check that all 3 metric keys are in the result."""
        model = train_linear(trained_data["X_train"], trained_data["y_train"])
        metrics = evaluate_model(model, trained_data["X_test"], trained_data["y_test"])

        expected_keys = ["mae", "rmse", "r2"]
        for key in expected_keys:
            assert key in metrics, f"Metric '{key}' is missing from results"

    def test_mae_and_rmse_non_negative(self, trained_data):
        """
        Check that MAE and RMSE are >= 0.

        MAE and RMSE measure error magnitude — they can never be negative.
        If they are, something is wrong with the calculation.
        """
        model = train_linear(trained_data["X_train"], trained_data["y_train"])
        metrics = evaluate_model(model, trained_data["X_test"], trained_data["y_test"])

        assert metrics["mae"] >= 0, f"MAE should be >= 0, got {metrics['mae']}"
        assert metrics["rmse"] >= 0, f"RMSE should be >= 0, got {metrics['rmse']}"

    def test_rmse_greater_or_equal_mae(self, trained_data):
        """
        Check that RMSE >= MAE.

        This is a mathematical property: RMSE always >= MAE because
        squaring amplifies large errors. If RMSE < MAE, the calculation
        is wrong.
        """
        model = train_linear(trained_data["X_train"], trained_data["y_train"])
        metrics = evaluate_model(model, trained_data["X_test"], trained_data["y_test"])

        assert metrics["rmse"] >= metrics["mae"], \
            f"RMSE ({metrics['rmse']}) should be >= MAE ({metrics['mae']})"

    def test_r2_is_float(self, trained_data):
        """
        Check that R² is a valid number.

        We don't check R² >= 0 because on random fake data,
        a model can perform worse than the mean (R² < 0).
        We only verify it's a real number, not NaN or inf.
        """
        model = train_linear(trained_data["X_train"], trained_data["y_train"])
        metrics = evaluate_model(model, trained_data["X_test"], trained_data["y_test"])

        assert isinstance(metrics["r2"], float), "R² should be a float"
        assert not np.isnan(metrics["r2"]), "R² should not be NaN"
        assert not np.isinf(metrics["r2"]), "R² should not be infinity"


# ============================================================
# TESTS FOR save_best_model()
# ============================================================

class TestSaveBestModel:
    """Tests for save_best_model() — same pattern as anomaly detector."""

    def test_save_creates_pkl(self, trained_data, tmp_path, monkeypatch):
        """Check that a .pkl model file is created and loadable."""
        pkl_path = tmp_path / "forecaster_model.pkl"
        json_path = tmp_path / "forecaster_metrics.json"

        monkeypatch.setattr("src.ml.trend_forecaster.MODEL_PATH", pkl_path)
        monkeypatch.setattr("src.ml.trend_forecaster.METRICS_PATH", json_path)

        model = train_linear(trained_data["X_train"], trained_data["y_train"])
        metrics = {"linear_regression": {"mae": 0.54}}

        save_best_model(model, "linear_regression", metrics, trained_data["scaler"])

        assert pkl_path.exists(), "Model .pkl file was not created"

        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        assert "model" in loaded, "Saved file missing 'model' key"
        assert "scaler" in loaded, "Saved file missing 'scaler' key"
        assert "name" in loaded, "Saved file missing 'name' key"

    def test_save_creates_json(self, trained_data, tmp_path, monkeypatch):
        """Check that a .json metrics file is created with valid content."""
        pkl_path = tmp_path / "forecaster_model.pkl"
        json_path = tmp_path / "forecaster_metrics.json"

        monkeypatch.setattr("src.ml.trend_forecaster.MODEL_PATH", pkl_path)
        monkeypatch.setattr("src.ml.trend_forecaster.METRICS_PATH", json_path)

        model = train_linear(trained_data["X_train"], trained_data["y_train"])
        metrics = {"linear_regression": {"mae": 0.54, "rmse": 0.87, "r2": 0.48}}

        save_best_model(model, "linear_regression", metrics, trained_data["scaler"])

        assert json_path.exists(), "Metrics .json file was not created"

        with open(json_path, "r") as f:
            loaded_metrics = json.load(f)

        assert "linear_regression" in loaded_metrics, \
            "Saved JSON missing 'linear_regression' key"