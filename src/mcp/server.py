"""
MCP Server — Model Context Protocol server for Claude Desktop.

What this module does:
- Exposes our ML models and data as "tools" for Claude Desktop
- Claude can call these tools directly during conversations
- No need to open a browser or API — just ask Claude

What is MCP?
Model Context Protocol is a standard by Anthropic that lets
AI assistants (like Claude) use external tools and data sources.
Think of it as "plugins for Claude".

How to run:
    python src/mcp/server.py

Then add to Claude Desktop config (see README for details).

Available tools:
- get_summary: campaign overview with KPIs
- get_anomalies: list detected anomalies
- get_campaign: data for a specific campaign
- get_importance: SHAP feature importance
- predict_roas: forecast ROAS for new data
- ask_question: RAG-powered natural language Q&A
"""

import json
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# MCP SDK
from mcp.server.fastmcp import FastMCP

# ============================================================
# SERVER SETUP
# ============================================================

# Create the MCP server instance
mcp = FastMCP("Ad Performance Intelligence")

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


# ============================================================
# HELPERS
# ============================================================

def load_features():
    """Load the features DataFrame."""
    if not FEATURES_PATH.exists():
        return None
    return pd.read_csv(FEATURES_PATH)


# ============================================================
# TOOLS
# ============================================================

@mcp.tool()
def get_summary() -> str:
    """
    Get a high-level summary of all ad campaign data.

    Returns overview with: total records, unique campaigns, date range,
    average ROAS/CTR/CPC, anomaly count, best and worst campaigns.
    """
    df = load_features()
    if df is None:
        return "Error: features.csv not found. Run the ETL pipeline first."

    avg_roas = df.groupby("campaign_name")["roas"].mean().sort_values()

    summary = {
        "total_rows": len(df),
        "unique_campaigns": int(df["campaign_name"].nunique()),
        "campaign_names": sorted(df["campaign_name"].unique().tolist()),
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
            "rate_percent": round(float(df["is_anomaly"].mean()) * 100, 1),
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

    return json.dumps(summary, indent=2)


@mcp.tool()
def get_anomalies(campaign_name: str = "") -> str:
    """
    Get all detected anomalies, optionally filtered by campaign name.

    Args:
        campaign_name: Filter by specific campaign (empty = all campaigns)

    Returns list of anomalous campaign entries with metrics.
    """
    df = load_features()
    if df is None:
        return "Error: features.csv not found."

    anomalies = df[df["is_anomaly"] == 1].copy()

    if campaign_name:
        anomalies = anomalies[anomalies["campaign_name"] == campaign_name]

    anomalies = anomalies.sort_values("c_date", ascending=False)

    display_cols = [
        "c_date", "campaign_name", "category", "roas", "ctr",
        "cpc", "roas_wow", "ctr_wow", "spend_wow",
    ]
    available_cols = [c for c in display_cols if c in anomalies.columns]

    result = {
        "count": len(anomalies),
        "total_rows": len(df),
        "anomaly_rate_percent": round(len(anomalies) / len(df) * 100, 1) if len(df) > 0 else 0,
        "anomalies": anomalies[available_cols].replace({np.nan: None}).to_dict(orient="records"),
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def get_campaign(campaign_name: str) -> str:
    """
    Get detailed performance data for a specific campaign.

    Args:
        campaign_name: Name of the campaign (e.g. 'facebook_tier1', 'google_hot')

    Returns all rows for the campaign with metrics and anomaly flags.
    """
    df = load_features()
    if df is None:
        return "Error: features.csv not found."

    campaign = df[df["campaign_name"] == campaign_name]

    if campaign.empty:
        available = sorted(df["campaign_name"].unique().tolist())
        return f"Campaign '{campaign_name}' not found. Available campaigns: {available}"

    summary = {
        "campaign_name": campaign_name,
        "category": campaign["category"].iloc[0],
        "total_entries": len(campaign),
        "date_range": {
            "start": str(campaign["c_date"].min()),
            "end": str(campaign["c_date"].max()),
        },
        "metrics": {
            "avg_roas": round(float(campaign["roas"].mean()), 4),
            "avg_ctr": round(float(campaign["ctr"].mean()), 4),
            "avg_cpc": round(float(campaign["cpc"].mean()), 4),
            "total_spent": round(float(campaign["mark_spent"].sum()), 2),
            "total_revenue": round(float(campaign["revenue"].sum()), 2),
        },
        "anomalies": int(campaign["is_anomaly"].sum()),
        "data": campaign.replace({np.nan: None}).to_dict(orient="records"),
    }

    return json.dumps(summary, indent=2)


@mcp.tool()
def get_importance() -> str:
    """
    Get SHAP-based feature importance for both models.

    Shows which features have the most influence on model predictions.
    Higher importance = more impact on decisions.
    """
    result = {}

    if SHAP_ANOMALY_PATH.exists():
        with open(SHAP_ANOMALY_PATH, "rb") as f:
            data = pickle.load(f)
        importance = np.abs(data["shap_values"]).mean(axis=0)
        result["anomaly_detector"] = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in sorted(
                zip(data["feature_names"], importance),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

    if SHAP_FORECASTER_PATH.exists():
        with open(SHAP_FORECASTER_PATH, "rb") as f:
            data = pickle.load(f)
        importance = np.abs(data["shap_values"]).mean(axis=0)
        result["roas_forecaster"] = [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in sorted(
                zip(data["feature_names"], importance),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

    if not result:
        return "No SHAP values found. Run the explainer first."

    return json.dumps(result, indent=2)


@mcp.tool()
def predict_roas(
    ctr: float,
    cpc: float,
    cpl: float,
    conversion_rate: float,
    ctr_wow: float,
    spend_wow: float,
    ctr_rolling3: float,
) -> str:
    """
    Predict ROAS for new campaign data.

    Args:
        ctr: Click-through rate (e.g. 0.04 = 4%)
        cpc: Cost per click in dollars (e.g. 2.50)
        cpl: Cost per lead in dollars (e.g. 12.00)
        conversion_rate: Orders/clicks ratio (e.g. 0.10 = 10%)
        ctr_wow: CTR week-over-week change (e.g. -0.20 = 20% drop)
        spend_wow: Spend week-over-week change (e.g. 0.15 = 15% increase)
        ctr_rolling3: 3-period rolling average CTR (e.g. 0.05)

    Returns predicted ROAS value and model info.
    """
    if not FORECASTER_MODEL_PATH.exists():
        return "Error: Forecaster model not found. Train models first."

    with open(FORECASTER_MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    scaler = model_data["scaler"]

    features = np.array([[ctr, cpc, cpl, conversion_rate, ctr_wow, spend_wow, ctr_rolling3]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    result = {
        "predicted_roas": round(float(prediction), 4),
        "model_used": model_data["name"],
        "input_data": {
            "ctr": ctr,
            "cpc": cpc,
            "cpl": cpl,
            "conversion_rate": conversion_rate,
            "ctr_wow": ctr_wow,
            "spend_wow": spend_wow,
            "ctr_rolling3": ctr_rolling3,
        },
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def ask_question(question: str) -> str:
    """
    Ask a natural language question about campaign performance.

    Uses RAG (Retrieval-Augmented Generation) to find relevant data
    and generate an answer powered by LLM.

    Args:
        question: Your question about campaigns (e.g. 'Which campaign has the worst ROAS?')

    Returns AI-generated answer based on real campaign data.
    """
    try:
        from src.rag.chain import ask as rag_ask
        result = rag_ask(question, top_k=5)

        response = {
            "question": question,
            "answer": result["answer"],
            "sources_count": len(result["sources"]),
            "source_campaigns": list(set(
                m.get("campaign_name", "") for m in result["metadatas"]
            )),
        }

        return json.dumps(response, indent=2)

    except Exception as e:
        return f"RAG pipeline error: {str(e)}"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
