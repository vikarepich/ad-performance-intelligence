"""
Trend Forecaster — predicts ROAS (Return on Ad Spend) for ad campaigns.

What this module does:
- Loads the feature-engineered dataset (features.csv)
- Trains 3 regression models: Linear Regression, XGBoost, LightGBM
- Evaluates each model using MAE, RMSE, R²
- Saves the best model to disk as a .pkl file
- Saves metrics for all models to a JSON file

Key difference from anomaly_detector.py:
- This is REGRESSION (predict a number), not classification (predict a class)
- Target: roas (continuous value), not is_anomaly (0 or 1)
- Metrics: MAE, RMSE, R² instead of precision, recall, f1

Input:  data/processed/features.csv
Output: models/forecaster_model.pkl, models/forecaster_metrics.json
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ============================================================
# PATHS
# ============================================================

FEATURES_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/forecaster_model.pkl")
METRICS_PATH = Path("models/forecaster_metrics.json")

# Features for ROAS prediction.
# IMPORTANT: we EXCLUDE roas, roas_wow, roas_rolling3
# because they are derived from the target variable (data leakage).
#
# What is data leakage?
# If we include roas-related features, the model "cheats" —
# it already knows the answer from the input.
# In production, we won't have future roas when making predictions.
FEATURE_COLUMNS = [
    "ctr",
    "cpc",
    "cpl",
    "conversion_rate",
    "ctr_wow",
    "spend_wow",
    "ctr_rolling3",
]

TARGET_COLUMN = "roas"


# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

def load_features():
    """Load the feature-engineered dataset."""
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} rows from {FEATURES_PATH}")
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Select features, split into train/test sets, and scale the data.

    Same approach as anomaly_detector.py, but:
    - No stratify (regression target is continuous, not classes)
    - Different feature set (no roas-derived columns)

    Parameters:
        df: DataFrame with features and target
        test_size: fraction of data for testing (0.2 = 20%)
        random_state: seed for reproducibility

    Returns:
        X_train, X_test: scaled feature matrices
        y_train, y_test: target arrays (roas values)
        scaler: fitted StandardScaler
    """
    # X = features, y = target (roas)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Replace any remaining NaN/inf with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split: 80% train, 20% test
    # No stratify here — roas is continuous, not categorical
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=FEATURE_COLUMNS,
        index=X_train.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=FEATURE_COLUMNS,
        index=X_test.index,
    )

    print(f"Train set: {len(X_train)} rows")
    print(f"Test set:  {len(X_test)} rows")
    print(f"ROAS mean in train: {y_train.mean():.2f}")
    print(f"ROAS mean in test:  {y_test.mean():.2f}")

    return X_train, X_test, y_train, y_test, scaler


# ============================================================
# MODEL TRAINING
# ============================================================

def train_linear(X_train, y_train):
    """
    Train Linear Regression — our baseline model.

    How it works:
    - Finds a straight line (hyperplane) that best fits the data
    - Formula: roas = w1*ctr + w2*cpc + ... + bias
    - "Best fit" means minimizing the sum of squared errors

    Why baseline?
    If a simple linear model predicts roas well, the relationship
    between features and roas is mostly linear — no need for
    complex models. If it fails, we know we need non-linear models.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Linear Regression trained")
    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost Regressor — gradient boosted trees for regression.

    Same algorithm as in anomaly_detector.py, but configured
    for regression instead of classification:
    - objective="reg:squarederror" (minimize squared error)
    - No eval_metric="logloss" (that's for classification)

    XGBoost often captures non-linear patterns that
    Linear Regression misses.
    """
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("XGBoost Regressor trained")
    return model


def train_lightgbm(X_train, y_train):
    """
    Train LightGBM Regressor — another gradient boosting algorithm.

    How LightGBM differs from XGBoost:
    - Uses "leaf-wise" tree growth (XGBoost uses "level-wise")
    - Faster training on large datasets
    - Often similar accuracy to XGBoost

    verbose=-1: suppress training logs (LightGBM is chatty by default)
    """
    model = LGBMRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    print("LightGBM Regressor trained")
    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, model_name="model"):
    """
    Evaluate a regression model and return a dictionary of metrics.

    Metrics explained:

    MAE (Mean Absolute Error):
        Average of |predicted - actual| for all predictions.
        Example: MAE=0.5 means "on average, predictions are off by 0.5"
        Easy to interpret, in the same units as the target (roas).

    RMSE (Root Mean Squared Error):
        Square root of the average squared errors.
        Penalizes large errors more than MAE.
        Example: if one prediction is off by 5.0, RMSE punishes it
        more than five predictions each off by 1.0.

    R² (R-squared / coefficient of determination):
        How much variance in roas the model explains.
        R²=1.0 — perfect predictions
        R²=0.0 — model is no better than always predicting the mean
        R²<0.0 — model is worse than predicting the mean (bad!)

    For MAE and RMSE: lower is better.
    For R²: higher is better (closer to 1.0).
    """
    y_pred = model.predict(X_test)

    metrics = {
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "r2": round(r2_score(y_test, y_pred), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  MAE:  {metrics['mae']}")
    print(f"  RMSE: {metrics['rmse']}")
    print(f"  R²:   {metrics['r2']}")

    return metrics


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(best_model, best_name, all_metrics, scaler):
    """
    Save the best model and scaler to disk.

    Same approach as anomaly_detector.py:
    - Model + scaler saved together as .pkl (pickle)
    - Metrics saved as .json for easy comparison
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save model + scaler together
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "name": best_name}, f)
    print(f"\nBest model '{best_name}' saved to {MODEL_PATH}")

    # Save metrics for all models
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_forecaster():
    """
    Full ROAS forecasting pipeline:
    1. Load data
    2. Prepare train/test sets
    3. Train all models
    4. Evaluate all models
    5. Pick the best one (by lowest MAE)
    6. Save to disk

    We pick the best model by MAE (not R²) because:
    - MAE is in the same units as roas (easy to interpret)
    - "Average error of 0.3 roas" is more actionable than "R²=0.85"
    """
    # Step 1: load data
    df = load_features()

    # Step 2: prepare train/test split
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Step 3: train all models
    models = {
        "linear_regression": train_linear(X_train, y_train),
        "xgboost": train_xgboost(X_train, y_train),
        "lightgbm": train_lightgbm(X_train, y_train),
    }

    # Step 4: evaluate all models
    all_metrics = {}
    for name, model in models.items():
        all_metrics[name] = evaluate_model(model, X_test, y_test, model_name=name)

    # Step 5: pick the best model by lowest MAE
    # min() because lower MAE = better predictions
    best_name = min(all_metrics, key=lambda name: all_metrics[name]["mae"])
    best_model = models[best_name]

    print(f"\n{'*'*50}")
    print(f"  WINNER: {best_name}")
    print(f"  MAE:  {all_metrics[best_name]['mae']}")
    print(f"  RMSE: {all_metrics[best_name]['rmse']}")
    print(f"  R²:   {all_metrics[best_name]['r2']}")
    print(f"{'*'*50}")

    # Step 6: save
    save_best_model(best_model, best_name, all_metrics, scaler)

    return models, all_metrics


if __name__ == "__main__":
    run_forecaster()