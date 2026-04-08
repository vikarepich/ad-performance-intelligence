"""
Tests for the Anomaly Detector (src/ml/anomaly_detector.py).

What's happening here:
- We test all key functions: prepare_data, train_*, evaluate_model, save_best_model
- We create a fake DataFrame in a fixture (same approach as test_pipeline.py)
- We use monkeypatch to redirect file saves to a temp folder

Key concept — why we test ML code:
ML code can "silently fail": no errors, but the model outputs garbage.
Tests catch issues like: wrong data shapes, NaN leaking into training,
metrics out of range, files not saving properly.
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier

from src.ml.anomaly_detector import (
    prepare_data,
    train_logistic,
    train_random_forest,
    train_xgboost,
    train_isolation_forest,
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
    Creates a fake DataFrame that mimics features.csv.

    We need enough rows for train_test_split to work:
    - at least 2 classes in both train and test
    - stratify requires minimum samples per class

    50 rows: 35 normal (0) + 15 anomalies (1) ≈ 30% anomaly rate
    (matches our real data distribution)

    np.random.seed(42) ensures the same random numbers every time —
    so tests are reproducible (same data = same results).
    """
    np.random.seed(42)
    n = 50

    data = {
        "ctr": np.random.uniform(0.01, 0.10, n),
        "cpc": np.random.uniform(0.5, 5.0, n),
        "roas": np.random.uniform(0.5, 8.0, n),
        "cpl": np.random.uniform(5.0, 50.0, n),
        "conversion_rate": np.random.uniform(0.01, 0.15, n),
        "roas_wow": np.random.uniform(-0.5, 0.5, n),
        "ctr_wow": np.random.uniform(-0.5, 0.5, n),
        "spend_wow": np.random.uniform(-0.5, 0.5, n),
        "roas_rolling3": np.random.uniform(1.0, 6.0, n),
        "ctr_rolling3": np.random.uniform(0.02, 0.08, n),
        "is_anomaly": [1] * 15 + [0] * 35,
    }

    return pd.DataFrame(data)


@pytest.fixture
def trained_data(sample_features_df):
    """
    Prepares train/test split from the fake data.

    This fixture DEPENDS on sample_features_df —
    pytest handles the chain automatically:
    sample_features_df runs first, then trained_data uses its result.

    Returns a dictionary with all the pieces we need for testing.
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
    """
    Tests for prepare_data().

    This function does 3 things:
    1. Selects feature columns and target
    2. Splits into train/test
    3. Scales features with StandardScaler
    """

    def test_returns_correct_shapes(self, sample_features_df):
        """Check that train/test sets have the right dimensions."""
        X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)

        assert len(X_train) + len(X_test) == len(sample_features_df), \
            "Train + test rows don't add up to original"

        assert X_train.shape[1] == len(FEATURE_COLUMNS), \
            f"Expected {len(FEATURE_COLUMNS)} features, got {X_train.shape[1]}"

        assert X_test.shape[1] == len(FEATURE_COLUMNS), \
            f"Expected {len(FEATURE_COLUMNS)} features, got {X_test.shape[1]}"

    def test_stratify_preserves_ratio(self, sample_features_df):
        """
        Check that anomaly ratio is similar in train and test.

        stratify=y in train_test_split ensures this.
        Without it, all anomalies could end up in one set by chance.
        We allow 10% tolerance because small datasets have natural variance.
        """
        X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)

        train_ratio = y_train.mean()
        test_ratio = y_test.mean()

        assert abs(train_ratio - test_ratio) < 0.10, \
            f"Anomaly ratios differ too much: train={train_ratio:.2f}, test={test_ratio:.2f}"

    def test_scaler_is_fitted(self, sample_features_df):
        """
        Check that the scaler has learned mean and std from training data.

        After fit_transform(), scaler.mean_ contains the mean of each feature.
        If this attribute exists, the scaler was properly fitted.
        """
        X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)

        assert hasattr(scaler, "mean_"), "Scaler was not fitted (no mean_)"
        assert len(scaler.mean_) == len(FEATURE_COLUMNS), \
            "Scaler mean_ length doesn't match feature count"

    def test_no_nan_after_preparation(self, sample_features_df):
        """Check that there are no NaN values after preparation."""
        X_train, X_test, y_train, y_test, scaler = prepare_data(sample_features_df)

        assert X_train.isna().sum().sum() == 0, "X_train contains NaN"
        assert X_test.isna().sum().sum() == 0, "X_test contains NaN"


# ============================================================
# TESTS FOR train functions
# ============================================================

class TestTrainModels:
    """
    Tests for each train_* function.

    We check 2 things for each model:
    1. The function returns the correct model type
    2. The model can make predictions (meaning it was fitted)

    hasattr(model, "predict") would pass even for unfitted models,
    so we actually CALL predict() to be sure.
    """

    def test_train_logistic(self, trained_data):
        """Check that Logistic Regression trains and predicts."""
        model = train_logistic(trained_data["X_train"], trained_data["y_train"])

        assert isinstance(model, LogisticRegression), \
            "Should return a LogisticRegression instance"

        predictions = model.predict(trained_data["X_test"])
        assert len(predictions) == len(trained_data["X_test"]), \
            "Prediction count doesn't match test set size"

    def test_train_random_forest(self, trained_data):
        """Check that Random Forest trains and predicts."""
        model = train_random_forest(trained_data["X_train"], trained_data["y_train"])

        assert isinstance(model, RandomForestClassifier), \
            "Should return a RandomForestClassifier instance"

        predictions = model.predict(trained_data["X_test"])
        assert len(predictions) == len(trained_data["X_test"]), \
            "Prediction count doesn't match test set size"

    def test_train_xgboost(self, trained_data):
        """Check that XGBoost trains and predicts."""
        model = train_xgboost(trained_data["X_train"], trained_data["y_train"])

        assert isinstance(model, XGBClassifier), \
            "Should return an XGBClassifier instance"

        predictions = model.predict(trained_data["X_test"])
        assert len(predictions) == len(trained_data["X_test"]), \
            "Prediction count doesn't match test set size"

    def test_train_isolation_forest(self, trained_data):
        """
        Check that Isolation Forest trains and predicts.

        Note: Isolation Forest only takes X_train (no y_train).
        Its predict() returns -1 (anomaly) or 1 (normal), not 0/1.
        """
        model = train_isolation_forest(trained_data["X_train"])

        assert isinstance(model, IsolationForest), \
            "Should return an IsolationForest instance"

        predictions = model.predict(trained_data["X_test"])
        assert len(predictions) == len(trained_data["X_test"]), \
            "Prediction count doesn't match test set size"

        unique_values = set(predictions)
        assert unique_values.issubset({-1, 1}), \
            f"Isolation Forest should predict -1 or 1, got {unique_values}"


# ============================================================
# TESTS FOR evaluate_model()
# ============================================================

class TestEvaluateModel:
    """
    Tests for evaluate_model().

    We check that:
    - All 4 metric keys are present
    - All metrics are in range [0, 1]
    - Isolation Forest is evaluated correctly (it has special handling)
    """

    def test_returns_all_metrics(self, trained_data):
        """Check that all 4 metric keys are in the result."""
        model = train_logistic(trained_data["X_train"], trained_data["y_train"])
        metrics = evaluate_model(model, trained_data["X_test"], trained_data["y_test"])

        expected_keys = ["accuracy", "precision", "recall", "f1"]
        for key in expected_keys:
            assert key in metrics, f"Metric '{key}' is missing from results"

    def test_metrics_in_valid_range(self, trained_data):
        """
        Check that all metrics are between 0.0 and 1.0.

        A metric outside this range means something is broken
        (e.g. wrong calculation, data leak, label mismatch).
        """
        model = train_logistic(trained_data["X_train"], trained_data["y_train"])
        metrics = evaluate_model(model, trained_data["X_test"], trained_data["y_test"])

        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, \
                f"Metric '{name}' = {value} is out of range [0, 1]"

    def test_evaluate_isolation_forest(self, trained_data):
        """
        Check that Isolation Forest evaluation works correctly.

        Isolation Forest returns -1/1 instead of 0/1.
        evaluate_model() should handle this conversion internally.
        The test ensures no errors and valid metrics.
        """
        model = train_isolation_forest(trained_data["X_train"])
        metrics = evaluate_model(model, trained_data["X_test"], trained_data["y_test"])

        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, \
                f"Isolation Forest metric '{name}' = {value} is out of range [0, 1]"


# ============================================================
# TESTS FOR save_best_model()
# ============================================================

class TestSaveBestModel:
    """
    Tests for save_best_model().

    We use monkeypatch to redirect saves to tmp_path
    so we don't overwrite the real model files.
    """

    def test_save_creates_pkl(self, trained_data, tmp_path, monkeypatch):
        """Check that a .pkl model file is created."""
        pkl_path = tmp_path / "anomaly_model.pkl"
        json_path = tmp_path / "anomaly_metrics.json"

        monkeypatch.setattr("src.ml.anomaly_detector.MODEL_PATH", pkl_path)
        monkeypatch.setattr("src.ml.anomaly_detector.METRICS_PATH", json_path)

        model = train_logistic(trained_data["X_train"], trained_data["y_train"])
        metrics = {"logistic_regression": {"f1": 0.79}}

        save_best_model(model, "logistic_regression", metrics, trained_data["scaler"])

        assert pkl_path.exists(), "Model .pkl file was not created"

        with open(pkl_path, "rb") as f:
            loaded = pickle.load(f)

        assert "model" in loaded, "Saved file missing 'model' key"
        assert "scaler" in loaded, "Saved file missing 'scaler' key"
        assert "name" in loaded, "Saved file missing 'name' key"

    def test_save_creates_json(self, trained_data, tmp_path, monkeypatch):
        """Check that a .json metrics file is created with valid content."""
        pkl_path = tmp_path / "anomaly_model.pkl"
        json_path = tmp_path / "anomaly_metrics.json"

        monkeypatch.setattr("src.ml.anomaly_detector.MODEL_PATH", pkl_path)
        monkeypatch.setattr("src.ml.anomaly_detector.METRICS_PATH", json_path)

        model = train_logistic(trained_data["X_train"], trained_data["y_train"])
        metrics = {"logistic_regression": {"accuracy": 0.88, "f1": 0.79}}

        save_best_model(model, "logistic_regression", metrics, trained_data["scaler"])

        assert json_path.exists(), "Metrics .json file was not created"

        with open(json_path, "r") as f:
            loaded_metrics = json.load(f)

        assert "logistic_regression" in loaded_metrics, \
            "Saved JSON missing 'logistic_regression' key"