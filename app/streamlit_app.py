"""
Streamlit Dashboard — Ad Performance Intelligence Engine.

What this module does:
- Shows campaign performance metrics and charts
- Displays anomalies with SHAP explanations
- Shows model performance metrics
- Provides an AI chat interface (RAG)

How to run:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
from pathlib import Path

# ============================================================
# PAGE CONFIG — must be first Streamlit command
# ============================================================

st.set_page_config(
    page_title="Ad Performance Intelligence",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# PATHS
# ============================================================

FEATURES_PATH = Path("data/processed/features.csv")
ANOMALY_METRICS_PATH = Path("models/anomaly_metrics.json")
FORECASTER_METRICS_PATH = Path("models/forecaster_metrics.json")
SHAP_ANOMALY_PATH = Path("models/shap_anomaly.pkl")
SHAP_FORECASTER_PATH = Path("models/shap_forecaster.pkl")


# ============================================================
# DATA LOADING — cached so it doesn't reload every interaction
# ============================================================

@st.cache_data
def load_data():
    """
    Load features.csv and cache it.

    @st.cache_data is a Streamlit decorator that caches the result.
    The function runs once, then returns the cached result on subsequent calls.
    This makes the dashboard fast — no reloading data on every click.
    """
    df = pd.read_csv(FEATURES_PATH)
    return df


@st.cache_data
def load_json_file(path):
    """Load a JSON file with caching."""
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data
def load_shap_file(path):
    """Load a SHAP pickle file with caching."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["shap_values"], data["feature_names"]


# ============================================================
# SIDEBAR — navigation
# ============================================================

st.sidebar.title("📊 Ad Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Campaign Performance", "Anomalies", "Model Metrics", "Feature Importance", "AI Chat"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with:** Python, scikit-learn, XGBoost, SHAP, ChromaDB, FastAPI, Streamlit")


# ============================================================
# LOAD DATA
# ============================================================

if not FEATURES_PATH.exists():
    st.error("features.csv not found. Run the ETL pipeline first: `python -m src.etl.pipeline`")
    st.stop()

df = load_data()


# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == "Overview":
    st.title("📊 Ad Performance Intelligence Engine")
    st.markdown("ML-powered analysis of ad campaign performance with anomaly detection, ROAS forecasting, and AI explanations.")

    # KPI cards — top-level metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Records",
            value=f"{len(df):,}",
        )

    with col2:
        st.metric(
            label="Campaigns",
            value=df["campaign_name"].nunique(),
        )

    with col3:
        anomaly_count = int(df["is_anomaly"].sum())
        anomaly_rate = round(anomaly_count / len(df) * 100, 1)
        st.metric(
            label="Anomalies",
            value=anomaly_count,
            delta=f"{anomaly_rate}% of total",
            delta_color="inverse",
        )

    with col4:
        st.metric(
            label="Avg ROAS",
            value=f"{df['roas'].mean():.2f}",
        )

    st.markdown("---")

    # Two charts side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("ROAS by Campaign")
        avg_roas = df.groupby("campaign_name")["roas"].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(
            avg_roas,
            x="roas",
            y="campaign_name",
            orientation="h",
            color="roas",
            color_continuous_scale="RdYlGn",
            labels={"roas": "Average ROAS", "campaign_name": "Campaign"},
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Anomaly Distribution")
        anomaly_by_campaign = df[df["is_anomaly"] == 1].groupby("campaign_name").size().sort_values(ascending=True).reset_index(name="anomalies")
        fig = px.bar(
            anomaly_by_campaign,
            x="anomalies",
            y="campaign_name",
            orientation="h",
            color="anomalies",
            color_continuous_scale="Reds",
            labels={"anomalies": "Anomaly Count", "campaign_name": "Campaign"},
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Date range info
    st.info(f"Data range: **{df['c_date'].min()}** to **{df['c_date'].max()}**")


# ============================================================
# PAGE: CAMPAIGN PERFORMANCE
# ============================================================

elif page == "Campaign Performance":
    st.title("📈 Campaign Performance")

    # Campaign filter
    campaigns = ["All"] + sorted(df["campaign_name"].unique().tolist())
    selected = st.selectbox("Select Campaign", campaigns)

    if selected == "All":
        filtered = df
    else:
        filtered = df[df["campaign_name"] == selected]

    # ROAS over time
    st.subheader("ROAS Over Time")
    fig = px.line(
        filtered,
        x="c_date",
        y="roas",
        color="campaign_name",
        markers=True,
        labels={"c_date": "Date", "roas": "ROAS", "campaign_name": "Campaign"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # CTR over time
    st.subheader("CTR Over Time")
    fig = px.line(
        filtered,
        x="c_date",
        y="ctr",
        color="campaign_name",
        markers=True,
        labels={"c_date": "Date", "ctr": "CTR", "campaign_name": "Campaign"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.subheader("Campaign Metrics Summary")
    summary = filtered.groupby("campaign_name").agg({
        "roas": "mean",
        "ctr": "mean",
        "cpc": "mean",
        "conversion_rate": "mean",
        "mark_spent": "sum",
        "revenue": "sum",
        "is_anomaly": "sum",
    }).round(4).reset_index()
    summary.columns = ["Campaign", "Avg ROAS", "Avg CTR", "Avg CPC", "Avg Conv Rate", "Total Spend", "Total Revenue", "Anomalies"]
    st.dataframe(summary, use_container_width=True)


# ============================================================
# PAGE: ANOMALIES
# ============================================================

elif page == "Anomalies":
    st.title("🚨 Anomaly Detection")

    anomalies = df[df["is_anomaly"] == 1].sort_values("c_date", ascending=False)

    st.metric("Total Anomalies", len(anomalies))

    # Filter by campaign
    campaigns = ["All"] + sorted(anomalies["campaign_name"].unique().tolist())
    selected = st.selectbox("Filter by Campaign", campaigns)

    if selected != "All":
        anomalies = anomalies[anomalies["campaign_name"] == selected]

    # Anomalies table
    st.subheader(f"Anomalies ({len(anomalies)} found)")
    display_cols = ["c_date", "campaign_name", "category", "roas", "ctr", "cpc", "roas_wow", "ctr_wow"]
    st.dataframe(
        anomalies[display_cols].reset_index(drop=True),
        use_container_width=True,
    )

    # SHAP explanations
    if SHAP_ANOMALY_PATH.exists():
        st.subheader("Why are these anomalies?")
        st.markdown("SHAP values show which features pushed each row toward being classified as an anomaly.")

        shap_values, feature_names = load_shap_file(str(SHAP_ANOMALY_PATH))

        # Show explanation for selected anomaly
        anomaly_indices = anomalies.index.tolist()
        if anomaly_indices:
            selected_idx = st.selectbox(
                "Select anomaly row to explain",
                anomaly_indices[:20],
                format_func=lambda x: f"Row {x} — {df.loc[x, 'campaign_name']} ({df.loc[x, 'c_date']})",
            )

            row_shap = shap_values[selected_idx]
            explanation_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": row_shap,
                "Abs SHAP": np.abs(row_shap),
            }).sort_values("Abs SHAP", ascending=False)

            # Bar chart of SHAP values
            fig = px.bar(
                explanation_df,
                x="SHAP Value",
                y="Feature",
                orientation="h",
                color="SHAP Value",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                labels={"SHAP Value": "Impact on Anomaly Prediction"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Text explanation
            top3 = explanation_df.head(3)
            st.markdown("**Top 3 factors:**")
            for _, row in top3.iterrows():
                direction = "TOWARD anomaly" if row["SHAP Value"] > 0 else "AWAY from anomaly"
                st.markdown(f"- **{row['Feature']}** (SHAP: {row['SHAP Value']:+.4f}) — pushes {direction}")


# ============================================================
# PAGE: MODEL METRICS
# ============================================================

elif page == "Model Metrics":
    st.title("🎯 Model Performance")

    col1, col2 = st.columns(2)

    # Anomaly Detector metrics
    with col1:
        st.subheader("Anomaly Detector")
        if ANOMALY_METRICS_PATH.exists():
            metrics = load_json_file(str(ANOMALY_METRICS_PATH))

            # Find best model
            best_name = max(metrics, key=lambda k: metrics[k]["f1"])

            for name, m in metrics.items():
                is_best = " 🏆" if name == best_name else ""
                with st.expander(f"{name}{is_best}", expanded=(name == best_name)):
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Accuracy", f"{m['accuracy']:.1%}")
                    mc2.metric("F1 Score", f"{m['f1']:.1%}")
                    mc1.metric("Precision", f"{m['precision']:.1%}")
                    mc2.metric("Recall", f"{m['recall']:.1%}")
        else:
            st.warning("No anomaly metrics found. Run anomaly detector first.")

    # ROAS Forecaster metrics
    with col2:
        st.subheader("ROAS Forecaster")
        if FORECASTER_METRICS_PATH.exists():
            metrics = load_json_file(str(FORECASTER_METRICS_PATH))

            # Find best model
            best_name = min(metrics, key=lambda k: metrics[k]["mae"])

            for name, m in metrics.items():
                is_best = " 🏆" if name == best_name else ""
                with st.expander(f"{name}{is_best}", expanded=(name == best_name)):
                    mc1, mc2 = st.columns(2)
                    mc1.metric("MAE", f"{m['mae']:.4f}")
                    mc2.metric("RMSE", f"{m['rmse']:.4f}")
                    mc1.metric("R²", f"{m['r2']:.4f}")
        else:
            st.warning("No forecaster metrics found. Run trend forecaster first.")


# ============================================================
# PAGE: FEATURE IMPORTANCE
# ============================================================

elif page == "Feature Importance":
    st.title("🔍 Feature Importance (SHAP)")
    st.markdown("SHAP values show how much each feature influences model predictions on average.")

    col1, col2 = st.columns(2)

    # Anomaly detector importance
    with col1:
        st.subheader("Anomaly Detector")
        if SHAP_ANOMALY_PATH.exists():
            shap_values, feature_names = load_shap_file(str(SHAP_ANOMALY_PATH))
            importance = np.abs(shap_values).mean(axis=0)

            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance,
            }).sort_values("Importance", ascending=True)

            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No SHAP values found. Run explainer first.")

    # Forecaster importance
    with col2:
        st.subheader("ROAS Forecaster")
        if SHAP_FORECASTER_PATH.exists():
            shap_values, feature_names = load_shap_file(str(SHAP_FORECASTER_PATH))
            importance = np.abs(shap_values).mean(axis=0)

            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance,
            }).sort_values("Importance", ascending=True)

            fig = px.bar(
                imp_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Greens",
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No SHAP values found. Run explainer first.")


# ============================================================
# PAGE: AI CHAT
# ============================================================

elif page == "AI Chat":
    st.title("💬 AI Campaign Analyst")
    st.markdown("Ask questions about your campaign data in natural language. Powered by RAG + Llama 3.1.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your campaigns..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching data and generating answer..."):
                try:
                    from src.rag.chain import ask as rag_ask
                    result = rag_ask(prompt, top_k=5)
                    answer = result["answer"]

                    st.markdown(answer)

                    # Show sources in expander
                    with st.expander("📄 Sources used"):
                        source_campaigns = [m.get("campaign_name", "") for m in result["metadatas"]]
                        st.markdown(f"**Campaigns referenced:** {', '.join(set(source_campaigns))}")
                        st.markdown(f"**Documents retrieved:** {len(result['sources'])}")

                except Exception as e:
                    answer = f"Error: {str(e)}"
                    st.error(answer)

        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Suggested questions
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        suggestions = [
            "Which campaign has the worst ROAS and why?",
            "Are there any anomalies in social media campaigns?",
            "What are the top performing campaigns?",
        ]
        for suggestion in suggestions:
            if st.button(suggestion, key=suggestion):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                st.rerun()