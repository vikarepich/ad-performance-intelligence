# 📊 Ad Performance Intelligence Engine

ML + RAG system that analyzes ad campaigns, detects anomalies, forecasts ROAS, and explains WHY your campaign broke — in plain English.

> Most BI tools tell you ROAS dropped. This one tells you **why** and **what to do about it**.

---

## The Problem

Small and mid-size businesses spend $2K+/month on dashboards that show them numbers they already know. Looker, Power BI, Tableau — they're great at visualization. But they don't:

- Detect anomalies automatically
- Explain *why* a metric changed
- Forecast what happens next
- Answer questions in natural language

This project does all four.

---

## What's Inside

### Phase 1: ETL Pipeline
- Ingests raw campaign data (308 rows, 11 columns)
- Engineers 25 features: CTR, CPC, ROAS, CPL, conversion rate, week-over-week changes, rolling averages, anomaly flags
- Detects 84 anomalies using rule-based thresholds (ROAS/CTR drop >20% vs rolling mean)

### Phase 2: ML Models
**Anomaly Detection** — 4 models benchmarked:
| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression 🏆 | 88% | 0.79 |
| XGBoost | 84% | 0.73 |
| Random Forest | 82% | 0.69 |
| Isolation Forest | 51% | 0.33 |

**ROAS Forecasting** — 3 models benchmarked:
| Model | MAE | R² |
|---|---|---|
| XGBoost 🏆 | 0.34 | 0.71 |
| LightGBM | 0.42 | 0.61 |
| Linear Regression | 0.54 | 0.48 |

**SHAP Explainability** — every prediction comes with "here's why":
- Anomaly detector: CTR (1.56), ctr_rolling3 (1.02), roas_wow (1.02) are top factors
- ROAS forecaster: conversion_rate (0.51), CPC (0.35), ctr_rolling3 (0.30) drive predictions

### Phase 3: RAG Layer
- **ChromaDB** vector database with 281 campaign documents
- **sentence-transformers/all-MiniLM-L6-v2** for embeddings (runs locally, no API needed)
- **Llama 3.1 8B** via HuggingFace Inference API for generation
- Ask "Which campaign has the worst ROAS?" → get an answer grounded in real data

### Phase 4: Product Layer
- **FastAPI** backend with 8 endpoints (`/campaigns`, `/anomalies`, `/predict`, `/ask`, etc.)
- **Streamlit** dashboard with 6 pages (Overview, Campaign Performance, Anomalies, Model Metrics, Feature Importance, AI Chat)
- **MCP Server** for Claude Desktop — 6 tools that let Claude query your campaigns directly

### Testing
- **112 pytest tests** across all modules
- Covers: ETL pipeline, anomaly detector, trend forecaster, SHAP explainer, RAG indexer/chain, FastAPI endpoints

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | pandas, numpy |
| ML | scikit-learn, XGBoost, LightGBM, SHAP |
| RAG | ChromaDB, sentence-transformers, HuggingFace Inference API |
| API | FastAPI, uvicorn |
| Dashboard | Streamlit, Plotly |
| Integration | MCP (Model Context Protocol) |
| Testing | pytest |
| Environment | Python 3.11, venv |

---

## Project Structure

```
ad-performance-intelligence/
├── data/
│   ├── raw/Marketing.csv          # Raw campaign data (308 rows)
│   └── processed/features.csv     # Engineered features (281 rows, 25 cols)
├── src/
│   ├── etl/pipeline.py            # ETL + feature engineering
│   ├── ml/
│   │   ├── anomaly_detector.py    # 4 classification models
│   │   ├── trend_forecaster.py    # 3 regression models
│   │   └── explainer.py           # SHAP explainability
│   ├── rag/
│   │   ├── indexer.py             # ChromaDB vector indexing
│   │   └── chain.py               # RAG pipeline (retrieve + generate)
│   ├── api/main.py                # FastAPI backend (8 endpoints)
│   └── mcp/server.py              # MCP server for Claude Desktop
├── app/streamlit_app.py           # Streamlit dashboard (6 pages)
├── models/                        # Saved models, metrics, SHAP values
├── tests/                         # 112 pytest tests
│   ├── test_pipeline.py
│   ├── test_anomaly_detector.py
│   ├── test_trend_forecaster.py
│   ├── test_explainer.py
│   ├── test_rag.py
│   └── test_api.py
├── requirements.txt
├── .env                           # HF_TOKEN (not committed)
└── .gitignore
```

---

## Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/vikarepich/ad-performance-intelligence.git
cd ad-performance-intelligence
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add API Token
```bash
# Create .env file with your HuggingFace token
echo "HF_TOKEN=hf_your_token_here" > .env
```
Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 3. Run the Pipeline
```bash
# Step 1: ETL
python -m src.etl.pipeline

# Step 2: Train models
python -m src.ml.anomaly_detector
python -m src.ml.trend_forecaster

# Step 3: SHAP explanations
python -m src.ml.explainer

# Step 4: Build vector index
python -m src.rag.indexer

# Step 5: Test RAG
python -m src.rag.chain
```

### 4. Run Tests
```bash
pytest tests/ -v
```

### 5. Launch Products
```bash
# FastAPI (http://localhost:8000/docs)
uvicorn src.api.main:app --reload --port 8000

# Streamlit (http://localhost:8501)
streamlit run app/streamlit_app.py

# MCP Server (for Claude Desktop)
python src/mcp/server.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/campaigns` | All campaigns (filterable) |
| GET | `/anomalies` | Detected anomalies |
| GET | `/metrics` | Model performance metrics |
| GET | `/importance` | SHAP feature importance |
| GET | `/summary` | High-level data overview |
| POST | `/ask` | RAG — ask a question |
| POST | `/predict` | Predict ROAS / detect anomaly |

---

## Roadmap

- [x] **Phase 1**: ETL pipeline + feature engineering
- [x] **Phase 2**: ML models (anomaly detection, ROAS forecasting, SHAP)
- [x] **Phase 3**: RAG layer (ChromaDB + LLM)
- [x] **Phase 4**: Product layer (FastAPI + Streamlit + MCP)
- [ ] **Phase 2.0**: Real API integrations (Google Ads, Meta Ads, GA4)
- [ ] **Phase 2.1**: PostHog for multi-touch attribution
- [ ] **Phase 2.2**: Self-serve product for SMBs

---

## Author

**Viktoria Repichinskaya**
Growth Leader Who Codes · 13+ years in healthtech, fintech, D2C

- [LinkedIn](https://www.linkedin.com/in/vikarepich/)
- [GitHub](https://github.com/vikarepich)