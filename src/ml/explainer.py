"""
Explainer — explains model predictions using SHAP values.

What this module does:
- Loads trained models (anomaly detector + trend forecaster)
- Computes SHAP values to explain WHY each prediction was made
- Identifies top contributing features for each prediction
- Generates human-readable explanations for anomalies
- Saves SHAP values for later use in Streamlit dashboard

What is SHAP?
SHAP (SHapley Additive exPlanations) comes from game theory.
Imagine 5 players won a game together — how much did each contribute?
SHAP answers the same question for ML features:
"How much did each feature contribute to THIS specific prediction?"

- Positive SHAP value = feature pushes prediction UP
- Negative SHAP value = feature pushes prediction DOWN
- Large absolute value = feature has strong influence

Input:  models/anomaly_model.pkl, models/forecaster_model.pkl, data/processed/features.csv
Output: models/shap_anomaly.pkl, models/shap_forecaster.pkl
"""

import pandas as pd
import numpy as np
import pickle
import shap
from pathlib import Path

from sklearn.ensemble import IsolationForest

# ============================================================
# PATHS
# ============================================================

FEATURES_PATH = Path("data/processed/features.csv")
ANOMALY_MODEL_PATH = Path("models/anomaly_model.pkl")
FORECASTER_MODEL_PATH = Path("models/forecaster_model.pkl")
SHAP_ANOMALY_PATH = Path("models/shap_anomaly.pkl")
SHAP_FORECASTER_PATH = Path("models/shap_forecaster.pkl")

# Feature sets (same as in anomaly_detector.py and trend_forecaster.py)
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
# LOAD MODEL
# ============================================================

def load_model(model_path):
    """
    Load a trained model + scaler from a .pkl file.

    The .pkl file contains a dictionary with keys:
    - "model": the trained sklearn/xgboost model
    - "scaler": the fitted StandardScaler
    - "name": the model name (e.g. "logistic_regression")

    Returns:
        model: the trained model object
        scaler: the fitted StandardScaler
        name: string name of the model
    """
    with open(model_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded model '{data['name']}' from {model_path}")
    return data["model"], data["scaler"], data["name"]


# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

def compute_shap_values(model, X, model_name="model"):
    """
    Compute SHAP values for all predictions.

    What happens inside:
    1. We create a SHAP explainer appropriate for the model type
    2. The explainer calculates how much each feature contributed
       to each prediction
    3. We get a matrix: rows = data points, columns = features

    Different models need different SHAP explainers:
    - TreeExplainer: for tree-based models (Random Forest, XGBoost, LightGBM)
      Fast and exact.
    - LinearExplainer: for linear models (Logistic Regression, Linear Regression)
      Uses the model coefficients directly.

    Parameters:
        model: trained model object
        X: DataFrame of scaled features
        model_name: string for logging

    Returns:
        shap_values: numpy array of shape (n_samples, n_features)
    """
    model_type = type(model).__name__

    # Choose the right SHAP explainer based on model type
    if model_type in ["RandomForestClassifier", "RandomForestRegressor",
                       "XGBClassifier", "XGBRegressor",
                       "LGBMClassifier", "LGBMRegressor"]:
        # TreeExplainer is fast and exact for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_result = explainer.shap_values(X)

        # For binary classification, TreeExplainer returns a list of 2 arrays
        # (one per class). We take class 1 (anomaly) values.
        if isinstance(shap_result, list):
            shap_values = shap_result[1]
        else:
            shap_values = shap_result

    elif model_type in ["LogisticRegression", "LinearRegression"]:
        # LinearExplainer works with linear models
        explainer = shap.LinearExplainer(model, X)
        shap_result = explainer.shap_values(X)

        # LinearExplainer for LogisticRegression may return a single array
        if isinstance(shap_result, list):
            shap_values = shap_result[1]
        else:
            shap_values = shap_result

    else:
        # KernelExplainer works with ANY model but is slow
        # We use a sample of data as background (for speed)
        background = shap.sample(X, min(50, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)

    print(f"SHAP values computed for {model_name}: shape {shap_values.shape}")
    return shap_values


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(shap_values, feature_names):
    """
    Calculate global feature importance from SHAP values.

    How it works:
    - For each feature, take the mean of absolute SHAP values
    - Higher mean |SHAP| = more important feature overall
    - Sort from most to least important

    This is GLOBAL importance (across all predictions).
    Unlike single-prediction explanations, this tells us
    which features matter MOST for the model in general.

    Parameters:
        shap_values: numpy array (n_samples, n_features)
        feature_names: list of feature column names

    Returns:
        DataFrame with columns: feature, importance (sorted descending)
    """
    # Mean absolute SHAP value per feature
    importance = np.abs(shap_values).mean(axis=0)

    # Create a sorted DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importance_df


def get_top_features(shap_values_row, feature_names, top_n=3):
    """
    Get the top N most influential features for a SINGLE prediction.

    This is LOCAL importance (for one specific row).
    Example output:
    [
        {"feature": "ctr_wow", "shap_value": 0.28, "rank": 1},
        {"feature": "roas_wow", "shap_value": 0.22, "rank": 2},
        {"feature": "spend_wow", "shap_value": -0.05, "rank": 3},
    ]

    Parameters:
        shap_values_row: 1D array of SHAP values for one prediction
        feature_names: list of feature column names
        top_n: how many top features to return

    Returns:
        list of dicts with feature name, SHAP value, and rank
    """
    # Sort by absolute SHAP value (highest impact first)
    indices = np.argsort(np.abs(shap_values_row))[::-1][:top_n]

    top = []
    for rank, idx in enumerate(indices, 1):
        top.append({
            "feature": feature_names[idx],
            "shap_value": round(float(shap_values_row[idx]), 4),
            "rank": rank,
        })

    return top


# ============================================================
# EXPLAIN ANOMALY
# ============================================================

def explain_anomaly(row_index, shap_values, X, feature_names, top_n=3):
    """
    Generate a human-readable explanation for a specific prediction.

    This is the most portfolio-worthy function:
    Instead of "row 42 is an anomaly", we get:
    "Row 42 is an anomaly because CTR dropped 35% (strong push toward anomaly)
     and ROAS dropped 42% (moderate push toward anomaly)"

    Parameters:
        row_index: which row to explain (index in the DataFrame)
        shap_values: full SHAP values matrix
        X: original feature DataFrame (scaled)
        feature_names: list of feature column names
        top_n: how many factors to include in explanation

    Returns:
        dict with row_index, top_factors, and text explanation
    """
    # Get SHAP values and feature values for this specific row
    row_shap = shap_values[row_index]
    row_features = X.iloc[row_index]

    # Get top contributing features
    top_factors = get_top_features(row_shap, feature_names, top_n)

    # Build human-readable explanation
    lines = [f"Row {row_index}:"]
    lines.append(f"  Top {top_n} factors:")

    for factor in top_factors:
        feature = factor["feature"]
        shap_val = factor["shap_value"]
        feat_val = round(float(row_features[feature]), 4)

        # Determine direction of influence
        if shap_val > 0:
            direction = "pushes TOWARD anomaly"
        else:
            direction = "pushes AWAY from anomaly"

        lines.append(
            f"  {factor['rank']}. {feature} = {feat_val} "
            f"(SHAP: {shap_val:+.4f}) — {direction}"
        )

    explanation = "\n".join(lines)

    return {
        "row_index": row_index,
        "top_factors": top_factors,
        "explanation": explanation,
    }


# ============================================================
# SAVE / LOAD SHAP VALUES
# ============================================================

def save_shap_values(shap_values, feature_names, output_path):
    """
    Save SHAP values to disk as a .pkl file.

    We save both shap_values and feature_names together
    so we can reconstruct explanations later.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump({
            "shap_values": shap_values,
            "feature_names": feature_names,
        }, f)

    print(f"SHAP values saved to {output_path}")


def load_shap_values(shap_path):
    """Load previously saved SHAP values."""
    with open(shap_path, "rb") as f:
        data = pickle.load(f)

    return data["shap_values"], data["feature_names"]


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_explainer():
    """
    Full explainability pipeline:
    1. Load data and models
    2. Scale data using saved scalers
    3. Compute SHAP values for both models
    4. Show global feature importance
    5. Explain top anomalies
    6. Save SHAP values for Streamlit
    """
    # Step 1: load data
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded {len(df)} rows from {FEATURES_PATH}")

    # Step 2: load anomaly model and compute SHAP
    print("\n" + "=" * 60)
    print("  ANOMALY DETECTOR EXPLANATIONS")
    print("=" * 60)

    anomaly_model, anomaly_scaler, anomaly_name = load_model(ANOMALY_MODEL_PATH)

    # Prepare data with the same scaler used during training
    X_anomaly = df[ANOMALY_FEATURES].copy()
    X_anomaly = X_anomaly.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_anomaly_scaled = pd.DataFrame(
        anomaly_scaler.transform(X_anomaly),
        columns=ANOMALY_FEATURES,
    )

    # Compute SHAP
    shap_anomaly = compute_shap_values(
        anomaly_model, X_anomaly_scaled, model_name=anomaly_name
    )

    # Global feature importance
    importance_anomaly = get_feature_importance(shap_anomaly, ANOMALY_FEATURES)
    print(f"\nGlobal feature importance ({anomaly_name}):")
    print(importance_anomaly.to_string(index=False))

    # Explain top 3 anomalies
    anomaly_indices = df[df["is_anomaly"] == 1].index[:3]
    print(f"\nExplanations for first 3 anomalies:")
    for idx in anomaly_indices:
        result = explain_anomaly(idx, shap_anomaly, X_anomaly_scaled, ANOMALY_FEATURES)
        print(f"\n{result['explanation']}")

    # Save SHAP values
    save_shap_values(shap_anomaly, ANOMALY_FEATURES, SHAP_ANOMALY_PATH)

    # Step 3: load forecaster model and compute SHAP
    print("\n" + "=" * 60)
    print("  ROAS FORECASTER EXPLANATIONS")
    print("=" * 60)

    forecaster_model, forecaster_scaler, forecaster_name = load_model(
        FORECASTER_MODEL_PATH
    )

    # Prepare data
    X_forecast = df[FORECASTER_FEATURES].copy()
    X_forecast = X_forecast.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_forecast_scaled = pd.DataFrame(
        forecaster_scaler.transform(X_forecast),
        columns=FORECASTER_FEATURES,
    )

    # Compute SHAP
    shap_forecast = compute_shap_values(
        forecaster_model, X_forecast_scaled, model_name=forecaster_name
    )

    # Global feature importance
    importance_forecast = get_feature_importance(shap_forecast, FORECASTER_FEATURES)
    print(f"\nGlobal feature importance ({forecaster_name}):")
    print(importance_forecast.to_string(index=False))

    # Save SHAP values
    save_shap_values(shap_forecast, FORECASTER_FEATURES, SHAP_FORECASTER_PATH)

    print("\n" + "*" * 60)
    print("  Explainer complete! SHAP values saved.")
    print("*" * 60)

    return {
        "anomaly": {
            "shap_values": shap_anomaly,
            "importance": importance_anomaly,
        },
        "forecaster": {
            "shap_values": shap_forecast,
            "importance": importance_forecast,
        },
    }


if __name__ == "__main__":
    run_explainer()