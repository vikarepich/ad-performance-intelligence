"""
Anomaly Detector — trains classification models to detect ad campaign anomalies.

What this module does:
- Loads the feature-engineered dataset (features.csv)
- Trains 4 models: Logistic Regression, Random Forest, XGBoost, Isolation Forest
- Evaluates each model using precision, recall, f1-score
- Saves the best model to disk as a .pkl file
- Saves metrics for all models to a JSON file

Input:  data/processed/features.csv
Output: models/anomaly_model.pkl, models/anomaly_metrics.json
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from xgboost import XGBClassifier

# ============================================================
# PATHS
# ============================================================

FEATURES_PATH = Path("data/processed/features.csv")
MODEL_PATH = Path("models/anomaly_model.pkl")
METRICS_PATH = Path("models/anomaly_metrics.json")

# Features we use for training.
# These are the numeric columns that describe campaign performance.
# We exclude: id, c_date, campaign_name, category, campaign_id (not numeric/useful),
# and is_anomaly (that's our target, not a feature).
# We also exclude intermediate columns: roas_prev, ctr_prev, spend_prev
# (they were only needed to calculate wow changes).
FEATURE_COLUMNS = [
    "ctr",
    "cpc",
    "roas",
    "cpl",
    "conversion_rate",
    "roas_wow",
    "ctr_wow",
    "spend_wow",
    "roas_rolling3",
    "ctr_rolling3",
]

TARGET_COLUMN = "is_anomaly"


# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

def load_features():
    """
    Load the feature-engineered dataset.

    Returns a DataFrame with all columns from features.csv.
    """
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} rows from {FEATURES_PATH}")
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Select features, split into train/test sets, and scale the data.

    What is train/test split?
    We divide data into 2 parts:
    - train (80%) — the model learns patterns from this data
    - test (20%) — we check how well the model works on NEW data it never saw

    What is StandardScaler?
    Different features have different scales (e.g. CTR is 0.01-0.10, CPC is 1-50).
    StandardScaler transforms each feature to have mean=0 and std=1.
    This helps models like Logistic Regression work better.

    Parameters:
        df: DataFrame with features and target
        test_size: fraction of data for testing (0.2 = 20%)
        random_state: seed for reproducibility (same split every time)

    Returns:
        X_train, X_test: scaled feature matrices
        y_train, y_test: target arrays
        scaler: fitted StandardScaler (needed later for new data)
    """
    # X = features (what the model sees), y = target (what it predicts)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Replace any remaining NaN/inf with 0 to avoid training errors
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split: 80% train, 20% test
    # stratify=y ensures the same proportion of anomalies in both sets
    # (important when one class is rare)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features to mean=0, std=1
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),  # fit on train data, then transform
        columns=FEATURE_COLUMNS,
        index=X_train.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),  # only transform (using train statistics!)
        columns=FEATURE_COLUMNS,
        index=X_test.index,
    )

    print(f"Train set: {len(X_train)} rows")
    print(f"Test set:  {len(X_test)} rows")
    print(f"Anomaly rate in train: {y_train.mean():.1%}")
    print(f"Anomaly rate in test:  {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test, scaler


# ============================================================
# MODEL TRAINING
# ============================================================

def train_logistic(X_train, y_train):
    """
    Train Logistic Regression — our baseline model.

    Why baseline?
    It's the simplest classification model. If a simple model works well,
    we might not need anything more complex. It also gives us a reference
    point: "can XGBoost beat Logistic Regression?"

    max_iter=1000: allow more iterations to converge (default 100 is often not enough).
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Logistic Regression trained")
    return model


def train_random_forest(X_train, y_train):
    """
    Train Random Forest — an ensemble of decision trees.

    How it works:
    - Creates 100 decision trees (n_estimators=100)
    - Each tree is trained on a random subset of data and features
    - Final prediction = majority vote of all trees

    Why it's good:
    - Handles non-linear relationships
    - Resistant to overfitting (averaging many trees)
    - Can tell us which features are most important
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,  # use all CPU cores for parallel training
    )
    model.fit(X_train, y_train)
    print("Random Forest trained")
    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost — gradient boosted trees.

    How it works:
    - Trains trees SEQUENTIALLY (not in parallel like Random Forest)
    - Each new tree corrects the errors of the previous one
    - Uses gradient descent to minimize the loss function

    Why it's good:
    - Often the top performer in tabular data competitions
    - Has built-in regularization to prevent overfitting
    - Handles missing values automatically

    eval_metric='logloss': binary classification loss function
    use_label_encoder=False: suppress deprecation warning
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("XGBoost trained")
    return model


def train_isolation_forest(X_train):
    """
    Train Isolation Forest — unsupervised anomaly detection.

    Key difference: this model does NOT use labels (y_train).
    It learns what "normal" data looks like and flags anything unusual.

    How it works:
    - Randomly selects a feature and a split point
    - Anomalies are easier to "isolate" (fewer splits needed)
    - Normal points need more splits to be isolated

    contamination=0.3: we expect ~30% of data to be anomalous
    (based on our ETL pipeline finding 84/281 ≈ 30% anomalies)

    Note: predict() returns -1 for anomaly, 1 for normal.
    We convert to 1/0 to match our target format.
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=0.3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)  # no y_train — unsupervised!
    print("Isolation Forest trained")
    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, model_name="model"):
    """
    Evaluate a model and return a dictionary of metrics.

    Metrics explained:
    - accuracy:  % of correct predictions overall
    - precision: when model says "anomaly", how often is it right?
                 (high precision = few false alarms)
    - recall:    of all real anomalies, how many did the model catch?
                 (high recall = we don't miss anomalies)
    - f1:        harmonic mean of precision and recall
                 (balanced metric, good for imbalanced data)

    Why not just accuracy?
    If 70% of data is normal, a model that always says "normal"
    gets 70% accuracy — but catches zero anomalies! F1 is better here.
    """
    # Isolation Forest returns -1/1 instead of 1/0
    if isinstance(model, IsolationForest):
        y_pred_raw = model.predict(X_test)
        # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        y_pred = (y_pred_raw == -1).astype(int)
    else:
        y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(best_model, best_name, all_metrics, scaler):
    """
    Save the best model and scaler to disk.

    What is pickle?
    A Python module that serializes (converts to bytes) any Python object.
    We save the model as a .pkl file so we can load it later without retraining.

    We also save the scaler because new data must be scaled the same way
    as training data (using the same mean and std).

    We save metrics as JSON so they're easy to read and compare.
    """
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save model + scaler together in one file
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "name": best_name}, f)
    print(f"\nBest model '{best_name}' saved to {MODEL_PATH}")

    # Save metrics for all models as JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_anomaly_detection():
    """
    Full anomaly detection pipeline:
    1. Load data
    2. Prepare train/test sets
    3. Train all models
    4. Evaluate all models
    5. Pick the best one (by f1-score)
    6. Save to disk
    """
    # Step 1: load data
    df = load_features()

    # Step 2: prepare train/test split
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    # Step 3: train all models
    models = {
        "logistic_regression": train_logistic(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
        "xgboost": train_xgboost(X_train, y_train),
        "isolation_forest": train_isolation_forest(X_train),
    }

    # Step 4: evaluate all models
    all_metrics = {}
    for name, model in models.items():
        all_metrics[name] = evaluate_model(model, X_test, y_test, model_name=name)

    # Step 5: pick the best model by f1-score
    best_name = max(all_metrics, key=lambda name: all_metrics[name]["f1"])
    best_model = models[best_name]

    print(f"\n{'*'*50}")
    print(f"  WINNER: {best_name}")
    print(f"  F1-score: {all_metrics[best_name]['f1']}")
    print(f"{'*'*50}")

    # Step 6: save
    save_best_model(best_model, best_name, all_metrics, scaler)

    return models, all_metrics


if __name__ == "__main__":
    run_anomaly_detection()