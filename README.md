# 📊 Ad Performance Intelligence Engine

ML + RAG system that analyzes ad campaigns, detects anomalies, forecasts ROAS, explains WHY your campaign broke, and solves multi-touch attribution — all in one place.

> Most BI tools tell you ROAS dropped. This one tells you **why**, **what to do about it**, and **which channel actually caused the conversion**.

---

## The Problem

Small and mid-size businesses spend $2K+/month on dashboards that show them numbers they already know. Looker, Power BI, Tableau — they're great at visualization. But they don't:

- Detect anomalies automatically
- Explain *why* a metric changed
- Forecast what happens next
- Answer questions in natural language
- Solve attribution (GA4 loses ~25% of touchpoints and uses last-click by default)

This project does all of the above.

---

## What's Inside

### Phase 1: ETL Pipeline
- Ingests raw campaign data from multiple sources
- Engineers 25 features: CTR, CPC, ROAS, CPL, conversion rate, week-over-week changes, rolling averages, anomaly flags
- Supports both CSV import and live connector data

### Phase 2: ML Models
**Anomaly Detection** — 4 models benchmarked:
| Model | Accuracy | F1 |
|---|---|---|
| Random Forest 🏆 | 91% | 0.91 |
| XGBoost | 91% | 0.91 |
| Logistic Regression | 79% | 0.81 |
| Isolation Forest | 49% | 0.47 |

**ROAS Forecasting** — 3 models benchmarked:
| Model | MAE | R² |
|---|---|---|
| LightGBM 🏆 | 0.18 | 0.90 |
| XGBoost | 0.21 | 0.84 |
| Linear Regression | 0.25 | 0.83 |

**SHAP Explainability** — every prediction comes with "here's why":
- Anomaly detector: roas_wow, roas, ctr_wow are top factors
- ROAS forecaster: conversion_rate, CPL, CPC drive predictions

### Phase 3: RAG Layer
- **ChromaDB** vector database with campaign documents
- **sentence-transformers/all-MiniLM-L6-v2** for embeddings (runs locally)
- **Llama 3.1 8B** via HuggingFace Inference API for generation
- Ask "Which campaign has the worst ROAS?" → get an answer grounded in real data

### Phase 4: Product Layer
- **FastAPI** backend with 8 endpoints (`/campaigns`, `/anomalies`, `/predict`, `/ask`, etc.)
- **Streamlit** dashboard with 6 pages (Overview, Campaign Performance, Anomalies, Model Metrics, Feature Importance, AI Chat)
- **MCP Server** for Claude Desktop — 6 tools that let Claude query your campaigns directly

### Phase 2.0: Multi-Platform Connectors
4 connectors with unified data format (mock data now, real API instructions included):
| Connector | Campaigns | What it tracks |
|---|---|---|
| **Google Ads** | Search brand, nonbrand, display retargeting | Impressions, clicks, cost, conversions |
| **Meta Ads** | Facebook feed, Instagram stories, lookalike | Impressions, clicks, spend, actions |
| **TikTok Ads** | In-feed video, TopView, Spark ads | Impressions, clicks, spend, conversions |
| **GA4** | By traffic source (google/cpc, facebook/paid, organic, direct) | Sessions, conversions, revenue |

Each connector includes step-by-step instructions for connecting real accounts.

### Phase 2.1: Multi-Touch Attribution
Solves the attribution problem that GA4 gets wrong:

**5 Attribution Models:**
| Model | How it works |
|---|---|
| Last-click | 100% credit to last touch (GA4 default — misleading) |
| First-click | 100% credit to first touch |
| Linear | Equal credit to all touches |
| Time-decay | More credit to recent touches |
| Shapley | Mathematically fair distribution (game theory) |

**Cookieless Tracking:**
- GA4 loses **24.8%** of touchpoints due to cookie rejection + ad blockers
- Enhanced tracking (UTM + fingerprint + server-side) recovers **22.3%** additional data
- Result: 97.5% visibility vs GA4's 75.2%

**Key Insight:** Last-click overvalues Google Brand Search by 5.5pp and undervalues TikTok by 3.6pp. Budget decisions based on last-click waste money.

### Testing
- **143 pytest tests** across all modules
- Covers: ETL, anomaly detector, trend forecaster, SHAP explainer, RAG, FastAPI, attribution

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
| Attribution | Custom Shapley implementation, cookieless tracking simulation |
| Testing | pytest (143 tests) |
| Environment | Python 3.11, venv |

---

## Project Structure

```
ad-performance-intelligence/
├── data/
│   ├── raw/
│   │   ├── Marketing.csv              # Original demo data
│   │   └── combined_campaigns.csv     # Multi-platform data (from connectors)
│   └── processed/
│       ├── features.csv               # Engineered features
│       ├── user_journeys.csv          # Simulated user journeys
│       └── attribution_results.csv    # Attribution model results
├── src/
│   ├── etl/pipeline.py                # ETL + feature engineering
│   ├── ml/
│   │   ├── anomaly_detector.py        # 4 classification models
│   │   ├── trend_forecaster.py        # 3 regression models
│   │   └── explainer.py               # SHAP explainability
│   ├── rag/
│   │   ├── indexer.py                 # ChromaDB vector indexing
│   │   └── chain.py                   # RAG pipeline (retrieve + generate)
│   ├── connectors/
│   │   ├── base.py                    # Unified connector interface
│   │   ├── google_ads.py              # Google Ads connector
│   │   ├── meta_ads.py               # Meta (Facebook/Instagram) connector
│   │   ├── tiktok_ads.py             # TikTok Ads connector
│   │   ├── ga4.py                     # Google Analytics 4 connector
│   │   └── manager.py                # Multi-source data collection
│   ├── attribution/
│   │   ├── journey_simulator.py       # User journey generation
│   │   ├── models.py                  # 5 attribution models
│   │   └── analyzer.py               # Model comparison + insights
│   ├── api/main.py                    # FastAPI backend (8 endpoints)
│   └── mcp/server.py                  # MCP server for Claude Desktop
├── app/streamlit_app.py               # Streamlit dashboard (6 pages)
├── models/                            # Saved models, metrics, SHAP values
├── tests/                             # 143 pytest tests
│   ├── test_pipeline.py
│   ├── test_anomaly_detector.py
│   ├── test_trend_forecaster.py
│   ├── test_explainer.py
│   ├── test_rag.py
│   ├── test_api.py
│   └── test_attribution.py
├── requirements.txt
├── .env                               # API tokens (not committed)
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
echo "HF_TOKEN=hf_your_token_here" > .env
```
Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 3. Run the Full Pipeline
```bash
# Collect data from all platforms (mock mode)
python -m src.connectors.manager

# ETL + feature engineering
python -m src.etl.pipeline

# Train ML models
python -m src.ml.anomaly_detector
python -m src.ml.trend_forecaster
python -m src.ml.explainer

# Build RAG index
python -m src.rag.indexer

# Run attribution analysis
python -m src.attribution.analyzer
```

### 4. Run Tests
```bash
pytest tests/ -v
# 143 tests, all passing
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
| GET | `/campaigns` | All campaigns (filterable by name, category, anomalies) |
| GET | `/anomalies` | Detected anomalies with details |
| GET | `/metrics` | Model performance metrics |
| GET | `/importance` | SHAP feature importance |
| GET | `/summary` | High-level data overview |
| POST | `/ask` | RAG — ask a question in natural language |
| POST | `/predict` | Predict ROAS / detect anomaly for new data |

---

## Attribution Results

Last-click attribution (GA4 default) vs Shapley (mathematically fair):

| Channel | Last-click | Shapley | Verdict |
|---|---|---|---|
| TikTok Paid | 10.9% | 14.5% | **Undervalued** by last-click |
| Meta Facebook | 11.0% | 14.3% | **Undervalued** by last-click |
| Google Brand Search | 20.7% | 15.2% | **Overvalued** by last-click |
| Direct | 12.1% | 11.2% | Fairly valued |

**Bottom line:** If you allocate budget based on last-click, you're overspending on Google Brand and underspending on TikTok and Meta — the channels that actually start the customer journey.

---

## Roadmap

- [x] Phase 1: ETL pipeline + feature engineering
- [x] Phase 2: ML models (anomaly detection, ROAS forecasting, SHAP)
- [x] Phase 3: RAG layer (ChromaDB + LLM)
- [x] Phase 4: Product layer (FastAPI + Streamlit + MCP)
- [x] Phase 2.0: Multi-platform connectors (Google, Meta, TikTok, GA4)
- [x] Phase 2.1: Multi-touch attribution (5 models + cookieless tracking)
- [ ] Phase 3.0: MLflow experiment tracking
- [ ] Phase 3.1: Docker containerization
- [ ] Phase 3.2: Real API integrations (live ad accounts)
- [ ] Phase 3.3: PostHog integration for production attribution
- [ ] Phase 3.4: Self-serve product for SMBs

---

## Author

**Viktoria Repichinskaya**
Growth Leader Who Codes · 13+ years in healthtech, fintech, D2C

- [LinkedIn](https://www.linkedin.com/in/vikarepich/)
- [GitHub](https://github.com/vikarepich)