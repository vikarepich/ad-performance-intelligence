# BigQuery + GA4 SQL Library

Production-grade SQL examples for analyzing **GA4 event data** in BigQuery. Six files covering the most common marketing analytics tasks: event unnesting, funnel analysis, cohort retention, multi-touch attribution, LTV by channel, and anomaly detection.

Built on the public dataset `bigquery-public-data.ga4_obfuscated_sample_ecommerce` — real GA4 export data from the Google Merchandise Store, January 2021.

---

## Why this exists

Most marketing analyst job descriptions in 2026 require **BigQuery + GA4 export** explicitly. The skill that separates juniors from seniors is not knowing SQL — it's knowing how to handle GA4's **nested ARRAY<STRUCT> schema**, multi-touch attribution logic, and the operational realities of production data (NULLs, sessionization, seasonality).

This library demonstrates all of that on real data.

---

## Files

| # | File | What it teaches |
|---|---|---|
| 01 | [`01_event_unnesting.sql`](ga4_examples/01_event_unnesting.sql) | UNNEST patterns for `event_params` (correlated subquery vs JOIN+pivot), traffic_source struct |
| 02 | [`02_funnel_analysis.sql`](ga4_examples/02_funnel_analysis.sql) | 4-step purchase funnel (view → cart → checkout → purchase) with conversion rates per step + by channel |
| 03 | [`03_cohort_retention.sql`](ga4_examples/03_cohort_retention.sql) | Daily cohort retention matrix using `_TABLE_SUFFIX` for date ranges |
| 04 | [`04_user_journey_attribution.sql`](ga4_examples/04_user_journey_attribution.sql) | Sessionization with forward-fill, multi-touch attribution (first/last/linear) |
| 05 | [`05_ltv_by_channel.sql`](ga4_examples/05_ltv_by_channel.sql) | 30-day LTV by acquisition channel with proxy metric for obfuscated data |
| 06 | [`06_anomaly_detection_sql.sql`](ga4_examples/06_anomaly_detection_sql.sql) | Three SQL-only anomaly detection methods (WoW, z-score, percentile) compared head-to-head |

---

## Key SQL techniques used

- `UNNEST(event_params)` — both correlated subquery and JOIN+`MAX(IF())` pivot patterns
- `_TABLE_SUFFIX` for wildcard table queries with date partition filtering (cost critical)
- Window functions: `ROW_NUMBER`, `LAG`, rolling `AVG`/`STDDEV` over preceding N rows, `MAX() OVER PARTITION BY` for forward-fill
- Multiple CTEs for staged transformation (sessionization → journey building → attribution scoring)
- `APPROX_QUANTILES` for percentile-based thresholds at scale
- `NULLIF` to safely guard divisions in conversion-rate calculations

---

## Sample insights surfaced from this data

Each file contains a real `INSIGHT:` block with findings from January 2021. Highlights:

**Funnel analysis (file 02):** Largest drop-off is view → add_to_cart (-82%). Cart → checkout (44%) and checkout → purchase (37%) are healthier than the top of the funnel — optimization priority should be on the product page, not the checkout flow.

**Multi-touch attribution (file 04):** GA4's default last-touch attribution **over-credits self-referral artifacts** (~+6% of total conversions) — payment redirect / OAuth flows mis-tracked as a "channel". Meanwhile, **all legitimate acquisition channels** (organic, real referrals) are systematically **under-credited** by last-touch vs linear.

**Direct vs Organic conversion (files 02, 05):** Direct traffic converts at **5.4%** while organic converts at **1.7%** — direct = users with formed brand intent. Performance-only marketers miss this: brand investment drives the highest-converting channel, not just paid acquisition.

**Anomaly detection methods (file 06):** Week-over-Week generated **6+ false positives** from a single seasonal shift (post-holiday shopping return). Z-score with a rolling baseline flagged only **1 strong anomaly + 1 suspicious** — its baseline adapts to new normal. Production systems combine multiple methods and alert only when 2+ agree.

---

## Data limitations

The public GA4 dataset is obfuscated for privacy: `item_revenue`, `price`, `item_id`, and `item_name` are **NULL or `(not set)`**. Where revenue would be relevant (LTV calculations), files use a **purchase-count proxy** with explicit comments showing how to swap in real revenue when running against production GA4 export.

This is intentional — handling imperfect data and documenting workarounds is itself a senior-level skill.

---

## How to run

1. Open [BigQuery Console](https://console.cloud.google.com/bigquery)
2. Pin the public dataset (or just reference it directly): `bigquery-public-data.ga4_obfuscated_sample_ecommerce`
3. Copy any query block from these files into a new SQL editor tab
4. Run — each file's queries are independent and runnable standalone

The Sandbox tier (free, no billing required) is enough — none of these queries scan more than a few GB.

---

## A note on VS Code red marks

If you have Microsoft SQL Server extensions (`ms-mssql.*`) installed in VS Code, you may see red error marks in these `.sql` files. The MSSQL parser doesn't recognize BigQuery-specific syntax (`UNNEST`, `_TABLE_SUFFIX`, backticks for table names, `STRUCT.field` access, `APPROX_QUANTILES`).

The queries are syntactically valid BigQuery SQL and execute correctly in the BigQuery Console. If the warnings annoy you, install [BigQuery Runner](https://marketplace.visualstudio.com/items?itemName=minodisk.bigquery-runner) — it provides BigQuery-aware syntax highlighting.

---

## Related modules in this project

- **`src/attribution/`** — Python implementations of Shapley and Markov chain attribution models (more sophisticated than the SQL versions here)
- **`src/ml/anomaly_detector.py`** — ML-based anomaly detection (Random Forest, XGBoost, LightGBM benchmark) for cases where SQL thresholds aren't enough
- **`src/connectors/ga4.py`** — GA4 connector using the same data shape patterns shown in these SQL files