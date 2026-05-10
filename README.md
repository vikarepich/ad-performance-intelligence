## Run with Docker (recommended — one command)

```bash
git clone https://github.com/vikarepich/ad-performance-intelligence.git
cd ad-performance-intelligence
echo "HF_TOKEN=hf_your_token_here" > .env  # free at huggingface.co/settings/tokens
docker-compose up -d
```

That's it. Two services come up with healthchecks:

- **FastAPI:** http://localhost:8000/docs (8 endpoints, OpenAPI docs)
- **Streamlit dashboard:** http://localhost:8501 (7 pages: Overview, Campaigns, Anomalies, Model Metrics, Feature Importance, AI Chat, Upload Data)

To stop: `docker-compose down`. To rebuild after code changes: `docker-compose build`.

---

## Run it locally

```bash
git clone https://github.com/vikarepich/ad-performance-intelligence.git
cd ad-performance-intelligence
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add a HuggingFace token (free) to `.env`:

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

Then run the full pipeline:

```bash
python -m src.connectors.manager       # Collect data from all platforms (mock mode)
python -m src.etl.pipeline             # Feature engineering
python -m src.ml.anomaly_detector      # Train ML models
python -m src.attribution.analyzer     # Run attribution analysis
python -m src.rag.indexer              # Build RAG index
pytest tests/ -v                       # 165 tests
```

---

<details>
<summary><strong>BigQuery + GA4 SQL library</strong> — 6 production-ready SQL files</summary>

Built on the public dataset `bigquery-public-data.ga4_obfuscated_sample_ecommerce` (real GA4 export from the Google Merchandise Store).

| File | Topic |
|---|---|
| `01_event_unnesting.sql` | UNNEST patterns for `event_params` (correlated subquery + JOIN+pivot), `traffic_source` struct |
| `02_funnel_analysis.sql` | 4-step purchase funnel + conversion rates + funnel-by-channel |
| `03_cohort_retention.sql` | Daily cohort retention matrix with `_TABLE_SUFFIX` |
| `04_user_journey_attribution.sql` | Sessionization with forward-fill + first/last/linear attribution |
| `05_ltv_by_channel.sql` | 30-day LTV by acquisition channel |
| `06_anomaly_detection_sql.sql` | WoW + z-score + percentile methods compared head-to-head |

Each file documents business questions, edge cases, and real insights from the data. See [bigquery/README.md](bigquery/README.md).

</details>

<details>
<summary><strong>ML model benchmarks</strong> — anomaly detection + ROAS forecasting</summary>

**Anomaly Detection (binary classification):**

| Model | Accuracy | F1 |
|---|---|---|
| Random Forest 🏆 | 91% | 0.91 |
| XGBoost | 91% | 0.91 |
| Logistic Regression | 79% | 0.81 |
| Isolation Forest | 49% | 0.47 |

**ROAS Forecasting (regression):**

| Model | MAE | R² |
|---|---|---|
| LightGBM 🏆 | 0.18 | 0.90 |
| XGBoost | 0.21 | 0.84 |
| Linear Regression | 0.25 | 0.83 |

**SHAP Explainability:** every prediction is paired with top contributing features. Anomaly detector's top drivers: `roas_wow`, `roas`, `ctr_wow`. Forecaster's top drivers: `conversion_rate`, `cpl`, `cpc`.

</details>

<details>
<summary><strong>Multi-touch attribution</strong> — 5 models + Markov chain</summary>

| Model | How it works |
|---|---|
| Last-click | 100% credit to last touch (GA4 default — biased) |
| First-click | 100% credit to first touch |
| Linear | Equal credit to all touches |
| Time-decay | More credit to recent touches |
| **Shapley** | Mathematically fair distribution (game theory) |
| **Markov chain** | Removal-effect probability model |

**Verification:** unit tests lock down two invariants:
1. **Revenue conservation** — all 6 models sum to the same total (€17,928.56)
2. **Markov-derived conversion probability** matches empirical rate to 6 decimal places

When Shapley and Markov **agree** on a directional disagreement with last-click, that's a high-confidence budget reallocation signal.

</details>

<details>
<summary><strong>Multi-platform connectors</strong> — Google Ads, Meta, TikTok, GA4</summary>

Unified connector interface, currently using mock data for portfolio purposes. Each connector includes:
- The unified data schema
- A working mock implementation
- Step-by-step instructions for real API integration

| Connector | Campaigns tracked |
|---|---|
| Google Ads | Search brand, nonbrand, display retargeting |
| Meta Ads | Facebook feed, Instagram stories, lookalike |
| TikTok Ads | In-feed video, TopView, Spark ads |
| GA4 | Sessions, conversions, revenue by source |

</details>

<details>
<summary><strong>RAG + product layer</strong> — ChromaDB + LLM, FastAPI, Streamlit, MCP</summary>

**RAG layer:**
- ChromaDB vector store with campaign documents
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings (runs locally)
- Llama 3.1 8B via HuggingFace Inference API for generation
- Ask questions like *"Which campaign has the worst ROAS?"* in natural language → grounded answer

**Products built on top:**
- **FastAPI** — 8 endpoints (`/campaigns`, `/anomalies`, `/predict`, `/ask`, etc.)
- **Streamlit** — 6-page dashboard (Overview, Campaigns, Anomalies, Model Metrics, Feature Importance, AI Chat)
- **MCP Server** — 6 tools that let Claude Desktop query campaigns directly

</details>

<details>
<summary><strong>Testing</strong> — 165 pytest tests</summary>

Covers ETL, anomaly detection, ROAS forecasting, SHAP explainer, RAG, FastAPI endpoints, and attribution models. Two key invariants are explicitly tested:

1. **Revenue conservation** across all 6 attribution models
2. **Markov-derived probability matches empirical conversion rate** to 6 decimal places

Run with: `pytest tests/ -v`

</details>

---

## Roadmap

- ✅ Phase 1: ETL pipeline + feature engineering
- ✅ Phase 2: ML models (anomaly detection, ROAS forecasting, SHAP)
- ✅ Phase 2.0: Multi-platform connectors
- ✅ Phase 2.1: Multi-touch attribution (5 models + cookieless tracking)
- ✅ Phase 3: RAG layer (ChromaDB + LLM)
- ✅ Phase 4: Product layer (FastAPI + Streamlit + MCP)
- ✅ Sprint 1.1: Markov chain attribution
- ✅ Sprint 1.2: BigQuery + GA4 SQL library (6 files)
- ⏳ Sprint 2: MLflow experiment tracking, Docker containerization, real API integrations

---

## About the author

**Viktoria Repich** — Marketing Data Analyst with growth + ML focus. 13 years across healthtech, fintech, and D2C. Currently building production-grade analytics tools that blend marketing intuition with engineering rigor.

- 🔗 [LinkedIn](https://www.linkedin.com/in/vik)
- 💻 [GitHub](https://github.com/repich-viktoriia)

## Documentation

- [Pitch Deck (PDF)](docs/ad_intelligence_pitch.pptx.pdf) — 8-slide project overview
- [User Manual (PDF)](docs/user_manual.docx.pdf) — full setup and usage guide