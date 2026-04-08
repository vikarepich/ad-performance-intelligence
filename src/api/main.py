"""
FastAPI Backend — REST API for the Ad Performance Intelligence Engine.

What this module does:
- Serves campaign data, anomalies, and model metrics via HTTP endpoints
- Provides SHAP-based feature importance explanations
- Accepts natural language questions via RAG pipeline
- Accepts new data for predictions

How to run:
    uvicorn src.api.main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive API documentation.

What is FastAPI?
A modern Python web framework for building APIs.
It's fast, auto-generates documentation, and validates input data.
You define functions with decorators (@app.get, @app.post),
and FastAPI turns them into HTTP endpoints.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# ============================================================
# APP SETUP
# ============================================================

# Create the FastAPI application
# title and description appear in the auto-generated docs at /docs
app = FastAPI(
    title="Ad Performance Intelligence API",
    description=(
        "ML-powered API for analyzing ad campaign performance. "
        "Detect anomalies, forecast ROAS, and get AI explanations."
    ),
    version="1.0.0",
)

# ============================================================
# PATHS
# ============================================================

FEATURES_PATH = Path("data/processed/features.csv")
ANOMALY_MODEL_PATH = Path("models/anomaly_model.pkl")
FORECASTER_MODEL_PATH = Path("models/forecaster_model.pkl")
ANOMALY_METRICS_PATH = Path("models/anomaly_metrics.json")
FORECASTER_METRICS_PATH = Path("models/forecaster_metrics.json")
SHAP_ANOMALY_PATH = Path("models/shap_anomaly.pkl")
SHAP_FORECASTER_PATH = Path("models/shap_forecaster.pkl")

# Feature columns (same as in ml modules)
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
# PYDANTIC MODELS — input validation
# ============================================================

class QuestionRequest(BaseModel):
    """
    Schema for the /ask endpoint.

    What is Pydantic?
    A library that validates input data automatically.
    If someone sends {"question": 123} instead of a string,
    FastAPI returns a clear error message.

    Example valid input:
    {"question": "Which campaign has the worst ROAS?", "top_k": 5}
    """
    question: str
    top_k: Optional[int] = 5


class PredictRequest(BaseModel):
    """
    Schema for the /predict endpoint.

    Accepts raw campaign metrics and returns predictions.
    All fields match the features our models expect.
    """
    ctr: float
    cpc: float
    cpl: float
    conversion_rate: float
    ctr_wow: float
    spend_wow: float
    ctr_rolling3: float
    # Additional fields for anomaly detection
    roas: Optional[float] = None
    roas_wow: Optional[float] = None
    roas_rolling3: Optional[float] = None


# ============================================================
# DATA LOADING HELPERS
# ============================================================

def load_features_df():
    """Load the features DataFrame."""
    if not FEATURES_PATH.exists():
        raise HTTPException(status_code=404, detail="features.csv not found. Run the ETL pipeline first.")
    return pd.read_csv(FEATURES_PATH)


def load_model(model_path):
    """Load a model + scaler from a .pkl file."""
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found at {model_path}. Train models first.")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_json(json_path):
    """Load a JSON metrics file."""
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"Metrics not found at {json_path}.")
    with open(json_path, "r") as f:
        return json.load(f)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health_check():
    """
    Health check — verify the API is running.

    This is a standard practice in production APIs.
    Monitoring tools ping /health to check if the service is alive.

    Returns: {"status": "ok"}
    """
    return {"status": "ok"}


@app.get("/campaigns")
def get_campaigns(
    campaign_name: Optional[str] = Query(None, description="Filter by campaign name"),
    category: Optional[str] = Query(None, description="Filter by category (social/search)"),
    anomalies_only: bool = Query(False, description="Show only anomalies"),
):
    """
    Get campaign data with optional filters.

    Query parameters allow filtering without changing the URL path:
    - /campaigns → all campaigns
    - /campaigns?campaign_name=facebook_tier1 → only facebook_tier1
    - /campaigns?anomalies_only=true → only anomalies
    - /campaigns?category=social&anomalies_only=true → social anomalies
    """
    df = load_features_df()

    # Apply filters
    if campaign_name:
        df = df[df["campaign_name"] == campaign_name]
    if category:
        df = df[df["category"] == category]
    if anomalies_only:
        df = df[df["is_anomaly"] == 1]

    # Convert to list of dicts (JSON-serializable)
    # replace NaN with None for clean JSON output
    records = df.replace({np.nan: None}).to_dict(orient="records")

    return {
        "count": len(records),
        "campaigns": records,
    }


@app.get("/anomalies")
def get_anomalies():
    """
    Get all detected anomalies with their details.

    Returns anomalies sorted by date (most recent first).
    """
    df = load_features_df()
    anomalies = df[df["is_anomaly"] == 1].sort_values("c_date", ascending=False)

    records = anomalies.replace({np.nan: None}).to_dict(orient="records")

    return {
        "count": len(records),
        "total_rows": len(df),
        "anomaly_rate": round(len(records) / len(df) * 100, 1),
        "anomalies": records,
    }


@app.get("/metrics")
def get_metrics():
    """
    Get model performance metrics.

    Returns metrics for both models:
    - Anomaly detector: accuracy, precision, recall, f1
    - ROAS forecaster: MAE, RMSE, R²
    """
    result = {}

    if ANOMALY_METRICS_PATH.exists():
        result["anomaly_detector"] = load_json(ANOMALY_METRICS_PATH)

    if FORECASTER_METRICS_PATH.exists():
        result["roas_forecaster"] = load_json(FORECASTER_METRICS_PATH)

    if not result:
        raise HTTPException(status_code=404, detail="No metrics found. Train models first.")

    return result


@app.get("/importance")
def get_feature_importance():
    """
    Get SHAP-based feature importance for both models.

    Shows which features have the most influence on predictions.
    Higher importance = more impact on model decisions.
    """
    result = {}

    if SHAP_ANOMALY_PATH.exists():
        with open(SHAP_ANOMALY_PATH, "rb") as f:
            shap_data = pickle.load(f)
        shap_values = shap_data["shap_values"]
        feature_names = shap_data["feature_names"]

        # Calculate mean absolute SHAP per feature
        importance = np.abs(shap_values).mean(axis=0)
        importance_list = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        result["anomaly_detector"] = importance_list

    if SHAP_FORECASTER_PATH.exists():
        with open(SHAP_FORECASTER_PATH, "rb") as f:
            shap_data = pickle.load(f)
        shap_values = shap_data["shap_values"]
        feature_names = shap_data["feature_names"]

        importance = np.abs(shap_values).mean(axis=0)
        importance_list = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        result["roas_forecaster"] = importance_list

    if not result:
        raise HTTPException(status_code=404, detail="No SHAP values found. Run the explainer first.")

    return result


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Ask a natural language question about campaign performance.

    Uses RAG (Retrieval-Augmented Generation):
    1. Finds relevant campaign data in ChromaDB
    2. Sends data + question to LLM
    3. Returns AI-generated answer with real data

    Example request:
    {"question": "Which campaign has the worst ROAS and why?", "top_k": 5}
    """
    try:
        from src.rag.chain import ask as rag_ask
        result = rag_ask(request.question, top_k=request.top_k)

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources_count": len(result["sources"]),
            "source_campaigns": [m.get("campaign_name", "") for m in result["metadatas"]],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Make predictions for new campaign data.

    Returns:
    - Anomaly prediction (if roas data provided)
    - ROAS forecast

    Example request:
    {
        "ctr": 0.04, "cpc": 2.5, "cpl": 12.0,
        "conversion_rate": 0.10, "ctr_wow": -0.20,
        "spend_wow": 0.15, "ctr_rolling3": 0.05,
        "roas": 3.0, "roas_wow": -0.30, "roas_rolling3": 3.5
    }
    """
    result = {}

    # --- ROAS Forecast ---
    try:
        forecaster_data = load_model(FORECASTER_MODEL_PATH)
        model = forecaster_data["model"]
        scaler = forecaster_data["scaler"]

        # Build feature array in correct order
        forecast_features = np.array([[
            request.ctr, request.cpc, request.cpl,
            request.conversion_rate, request.ctr_wow,
            request.spend_wow, request.ctr_rolling3,
        ]])

        # Scale and predict
        forecast_scaled = scaler.transform(forecast_features)
        roas_prediction = model.predict(forecast_scaled)[0]

        result["roas_forecast"] = {
            "predicted_roas": round(float(roas_prediction), 4),
            "model_used": forecaster_data["name"],
        }
    except HTTPException:
        result["roas_forecast"] = {"error": "Forecaster model not found"}
    except Exception as e:
        result["roas_forecast"] = {"error": str(e)}

    # --- Anomaly Detection ---
    if request.roas is not None and request.roas_wow is not None and request.roas_rolling3 is not None:
        try:
            anomaly_data = load_model(ANOMALY_MODEL_PATH)
            model = anomaly_data["model"]
            scaler = anomaly_data["scaler"]

            anomaly_features = np.array([[
                request.ctr, request.cpc, request.roas, request.cpl,
                request.conversion_rate, request.roas_wow, request.ctr_wow,
                request.spend_wow, request.roas_rolling3, request.ctr_rolling3,
            ]])

            anomaly_scaled = scaler.transform(anomaly_features)
            prediction = model.predict(anomaly_scaled)[0]

            result["anomaly_detection"] = {
                "is_anomaly": bool(prediction),
                "label": "ANOMALY" if prediction else "NORMAL",
                "model_used": anomaly_data["name"],
            }
        except HTTPException:
            result["anomaly_detection"] = {"error": "Anomaly model not found"}
        except Exception as e:
            result["anomaly_detection"] = {"error": str(e)}
    else:
        result["anomaly_detection"] = {
            "skipped": "Provide roas, roas_wow, roas_rolling3 for anomaly detection"
        }

    return result


@app.get("/summary")
def get_summary():
    """
    Get a high-level summary of all campaign data.

    Useful for dashboard overview:
    - Total campaigns, date range
    - Average metrics across all campaigns
    - Anomaly count and rate
    - Best and worst performers
    """
    df = load_features_df()

    # Best and worst campaigns by average ROAS
    avg_roas = df.groupby("campaign_name")["roas"].mean().sort_values()

    summary = {
        "total_rows": len(df),
        "unique_campaigns": int(df["campaign_name"].nunique()),
        "date_range": {
            "start": str(df["c_date"].min()),
            "end": str(df["c_date"].max()),
        },
        "average_metrics": {
            "roas": round(float(df["roas"].mean()), 4),
            "ctr": round(float(df["ctr"].mean()), 4),
            "cpc": round(float(df["cpc"].mean()), 4),
            "conversion_rate": round(float(df["conversion_rate"].mean()), 4),
        },
        "anomalies": {
            "count": int(df["is_anomaly"].sum()),
            "rate": round(float(df["is_anomaly"].mean()) * 100, 1),
        },
        "best_campaign": {
            "name": avg_roas.index[-1],
            "avg_roas": round(float(avg_roas.iloc[-1]), 4),
        },
        "worst_campaign": {
            "name": avg_roas.index[0],
            "avg_roas": round(float(avg_roas.iloc[0]), 4),
        },
    }

    return summary