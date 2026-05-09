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

**6 Attribution Models:**
| Model | How it works | Cost |
|---|---|---|
| Last-click | 100% credit to last touch (GA4 default — misleading) | O(1) |
| First-click | 100% credit to first touch | O(1) |
| Linear | Equal credit to all touches | O(N) |
| Time-decay | More credit to recent touches | O(N) |
| Shapley | Cooperative game theory — credit by marginal contribution across all coalitions | O(2^N) |
| Markov chain | Path-aware — credit by removal effect (drop in conversion probability when channel is turned off) | O(N²) |

Shapley and Markov are independent data-driven methods: Shapley is order-agnostic, Markov is path-aware. When they agree directionally, that's a high-confidence signal for budget reallocation.


**Cookieless Tracking:**
- GA4 loses **24.8%** of touchpoints due to cookie rejection + ad blockers
- Enhanced tracking (UTM + fingerprint + server-side) recovers **22.3%** additional data
- Result: 97.5% visibility vs GA4's 75.2%

**Key insight:** Both data-driven models agree that last-click overvalues Google Brand Search (by 3-5pp) and undervalues TikTok and Meta Facebook (by 2-4pp each). Two independent methodologies pointing the same direction = high-confidence signal that budget allocated on last-click is being wasted on bottom-funnel and starved from top-funnel.

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
| Attribution | Custom Shapley + Markov chain implementations, cookieless tracking simulation |
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
│   │   ├── models.py                  # 5 rule-based + Shapley attribution models
│   │   ├── markov_model.py            # Markov chain attribution (path-aware, removal effect)
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

Last-click vs two independent data-driven methods (€17,928 total revenue, 2,000 user journeys):

| Channel | Last-click | Markov | Shapley | Verdict |
|---|---|---|---|---|
| Google Brand Search | 20.7% | 17.4% | 15.2% | **Overvalued** by last-click (both models agree) |
| Google Search Nonbrand | 15.0% | 14.6% | 15.8% | Fairly valued |
| TikTok Paid | 10.9% | 13.9% | 14.5% | **Undervalued** by last-click |
| Meta Facebook | 11.0% | 13.2% | 14.3% | **Undervalued** by last-click |
| Direct | 12.1% | 10.8% | 11.2% | Fairly valued |
| Google Display Retargeting | 9.7% | 10.3% | 9.8% | Fairly valued |
| Organic Search | 11.2% | 10.0% | 8.4% | Slightly overvalued |
| Meta Instagram Stories | 9.4% | 9.9% | 10.8% | Fairly valued |

**Bottom line:** Markov and Shapley independently arrive at the same conclusion — last-click overspends on Google Brand Search and starves the top-of-funnel channels (TikTok, Meta) that actually start the customer journey. On €1M/month spend, the Google Brand overcrediting alone amounts to ~€33K/month misallocated.

**Why two data-driven methods, not one?** They use different math: Shapley evaluates channels through cooperative game theory (marginal contribution across coalitions, O(2^N)). Markov models journeys as a state graph and computes "removal effect" (drop in conversion probability if a channel is turned off, O(N²)). Both are derived from the same journey data; agreement between them is an independent verification, not luck.

All six models conserve total revenue at €17,928.56 — a sanity-check invariant verified by pytest.

---

## Roadmap

- [x] Phase 1: ETL pipeline + feature engineering
- [x] Phase 2: ML models (anomaly detection, ROAS forecasting, SHAP)
- [x] Phase 3: RAG layer (ChromaDB + LLM)
- [x] Phase 4: Product layer (FastAPI + Streamlit + MCP)
- [x] Phase 2.0: Multi-platform connectors (Google, Meta, TikTok, GA4)
- [x] Phase 2.1: Multi-touch attribution (6 models including Shapley + Markov, cookieless tracking)
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

---

## Documentation

- [Pitch Deck (PDF)](docs/ad_intelligence_pitch.pptx.pdf) — 8-slide project overview
- [User Manual (PDF)](docs/user_manual.docx.pdf) — full setup and usage guide

---