"""
MLflow Tracking — logs all ML experiments for comparison and reproducibility.

What this module does:
- Wraps anomaly_detector and trend_forecaster with MLflow tracking
- Logs hyperparameters, metrics, and models for every training run
- Provides a UI to compare experiments: mlflow ui

What is MLflow?
A platform for managing the ML lifecycle:
- Tracking: log parameters, metrics, artifacts for each experiment
- Models: save and version models
- UI: visual dashboard to compare runs

Why we use it:
Without MLflow, you lose track of experiments:
"Was it XGBoost with max_depth=4 or 6 that gave F1=0.91?"
MLflow saves everything automatically.

How to view results:
    mlflow ui --port 5050
    # Open http://localhost:5050

How to run experiments:
    python -m src.ml.mlflow_tracking
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor

# ============================================================
# CONFIG
# ============================================================

FEATURES_PATH = Path("data/processed/features.csv")

# MLflow experiment names
ANOMALY_EXPERIMENT = "anomaly_detection"
FORECASTER_EXPERIMENT = "roas_forecasting"

# Feature columns (same as in the ML modules)
ANOMALY_FEATURES = [
    "ctr", "cpc", "roas", "cpl", "conversion_rate",
    "roas_wow", "ctr_wow", "spend_wow",
    "roas_rolling3", "ctr_rolling3",
]

FORECASTER_FEATURES = [
    "ctr", "cpc", "cpl", "conversion_rate",
    "ctr_wow", "spend_wow", "ctr_rolling3",
]


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_prepare(feature_columns, target_column, stratify=True):
    """Load data and prepare train/test split."""
    df = pd.read_csv(FEATURES_PATH)

    X = df[feature_columns].copy()
    y = df[target_column].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    split_params = {
        "test_size": 0.2,
        "random_state": 42,
    }
    if stratify:
        split_params["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_params)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_columns,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================
# ANOMALY DETECTION EXPERIMENTS
# ============================================================

def run_anomaly_experiments():
    """
    Train all anomaly detection models with MLflow tracking.

    Each model gets its own MLflow "run" with:
    - Parameters: model type, hyperparameters
    - Metrics: accuracy, precision, recall, f1
    - Artifacts: the trained model itself
    """
    mlflow.set_experiment(ANOMALY_EXPERIMENT)

    X_train, X_test, y_train, y_test, scaler = load_and_prepare(
        ANOMALY_FEATURES, "is_anomaly", stratify=True
    )

    print(f"\n{'=' * 60}")
    print(f"  ANOMALY DETECTION — MLflow Tracking")
    print(f"{'=' * 60}")
    print(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"  Anomaly rate: {y_train.mean():.1%}")

    # --- Logistic Regression ---
    with mlflow.start_run(run_name="logistic_regression"):
        params = {"max_iter": 1000, "random_state": 42}
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LogisticRegression")

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = log_classification_metrics(y_test, y_pred)
        mlflow.sklearn.log_model(model, "model")
        print(f"  Logistic Regression: F1={metrics['f1']:.4f}")

    # --- Random Forest ---
    with mlflow.start_run(run_name="random_forest"):
        params = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForest")

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = log_classification_metrics(y_test, y_pred)
        mlflow.sklearn.log_model(model, "model")
        print(f"  Random Forest: F1={metrics['f1']:.4f}")

    # --- XGBoost ---
    with mlflow.start_run(run_name="xgboost"):
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        mlflow.log_params(params)
        mlflow.log_param("model_type", "XGBoost")

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = log_classification_metrics(y_test, y_pred)
        mlflow.xgboost.log_model(model, "model")
        print(f"  XGBoost: F1={metrics['f1']:.4f}")

    # --- Isolation Forest ---
    with mlflow.start_run(run_name="isolation_forest"):
        params = {
            "n_estimators": 100,
            "contamination": 0.3,
            "random_state": 42,
            "n_jobs": -1,
        }
        mlflow.log_params(params)
        mlflow.log_param("model_type", "IsolationForest")

        model = IsolationForest(**params)
        model.fit(X_train)
        y_pred_raw = model.predict(X_test)
        y_pred = (y_pred_raw == -1).astype(int)

        metrics = log_classification_metrics(y_test, y_pred)
        mlflow.sklearn.log_model(model, "model")
        print(f"  Isolation Forest: F1={metrics['f1']:.4f}")


# ============================================================
# ROAS FORECASTING EXPERIMENTS
# ============================================================

def run_forecaster_experiments():
    """
    Train all ROAS forecasting models with MLflow tracking.

    Each model gets its own MLflow "run" with:
    - Parameters: model type, hyperparameters
    - Metrics: MAE, RMSE, R²
    - Artifacts: the trained model
    """
    mlflow.set_experiment(FORECASTER_EXPERIMENT)

    X_train, X_test, y_train, y_test, scaler = load_and_prepare(
        FORECASTER_FEATURES, "roas", stratify=False
    )

    print(f"\n{'=' * 60}")
    print(f"  ROAS FORECASTING — MLflow Tracking")
    print(f"{'=' * 60}")
    print(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"  ROAS mean: {y_train.mean():.2f}")

    # --- Linear Regression ---
    with mlflow.start_run(run_name="linear_regression"):
        mlflow.log_param("model_type", "LinearRegression")

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = log_regression_metrics(y_test, y_pred)
        mlflow.sklearn.log_model(model, "model")
        print(f"  Linear Regression: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

    # --- XGBoost Regressor ---
    with mlflow.start_run(run_name="xgboost_regressor"):
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
        }
        mlflow.log_params(params)
        mlflow.log_param("model_type", "XGBoostRegressor")

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = log_regression_metrics(y_test, y_pred)
        mlflow.xgboost.log_model(model, "model")
        print(f"  XGBoost Regressor: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

    # --- LightGBM ---
    with mlflow.start_run(run_name="lightgbm_regressor"):
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LightGBM")

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = log_regression_metrics(y_test, y_pred)
        mlflow.sklearn.log_model(model, "model")
        print(f"  LightGBM: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")


# ============================================================
# METRIC LOGGING HELPERS
# ============================================================

def log_classification_metrics(y_test, y_pred):
    """Calculate and log classification metrics to MLflow."""
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    mlflow.log_metrics(metrics)
    return metrics


def log_regression_metrics(y_test, y_pred):
    """Calculate and log regression metrics to MLflow."""
    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2": round(r2_score(y_test, y_pred), 4),
    }
    mlflow.log_metrics(metrics)
    return metrics


# ============================================================
# MAIN
# ============================================================

def run_all_experiments():
    """
    Run all ML experiments with MLflow tracking.

    After running, view results with:
        mlflow ui --port 5050
        # Open http://localhost:5050
    """
    print("=" * 60)
    print("  MLflow EXPERIMENT TRACKING")
    print("=" * 60)

    # Set tracking URI to local folder
    mlflow.set_tracking_uri("mlruns")

    run_anomaly_experiments()
    run_forecaster_experiments()

    print(f"\n{'*' * 60}")
    print("  All experiments logged to MLflow!")
    print("  View results: mlflow ui --port 5050")
    print("  Then open: http://localhost:5050")
    print(f"{'*' * 60}")


if __name__ == "__main__":
    run_all_experiments()