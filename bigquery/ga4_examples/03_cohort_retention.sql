-- =====================================================================
-- 03_cohort_retention.sql
-- =====================================================================
-- BUSINESS QUESTION:
-- "Of users who first visited on day X, how many came back on day X+1, 
--  X+2, ..., X+7? Is product retention healthy?"
--
-- WHY THIS MATTERS:
-- Retention is the #1 product health metric. Acquisition without 
-- retention = leaky bucket. The classic cohort matrix is required 
-- knowledge for any analyst interview at a product company.
--
-- KEY CONCEPT — _TABLE_SUFFIX:
-- GA4 stores data as one table per day (events_20210101, events_20210102...).
-- The events_* wildcard + _TABLE_SUFFIX filter lets you query a date range
-- without scanning all tables (performance + cost critical).
--
-- DATASET: bigquery-public-data.ga4_obfuscated_sample_ecommerce
-- DATE RANGE: 2021-01-01 to 2021-01-31
-- =====================================================================


-- ---------------------------------------------------------------------
-- QUERY 1: Daily cohort sizes (smoke test)
-- ---------------------------------------------------------------------
-- For each day, count users whose FIRST EVER visit was that day.
-- This is the "Day 0" of each cohort — the denominator for retention.
--
-- WHY first_visit event: GA4 fires a special 'first_visit' event the 
-- first time a user is seen. We use it to identify cohort assignment.
-- ---------------------------------------------------------------------

SELECT
  PARSE_DATE('%Y%m%d', _TABLE_SUFFIX) AS cohort_date,
  COUNT(DISTINCT user_pseudo_id) AS new_users
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210131'
  AND event_name = 'first_visit'
GROUP BY cohort_date
ORDER BY cohort_date
LIMIT 10;

-- Expected output:
-- cohort_date | new_users
-- 2021-01-01  | ~2000
-- 2021-01-02  | ~2500
-- ... (each day's "new acquisition" volume)


-- ---------------------------------------------------------------------
-- QUERY 2: Cohort retention matrix (THE CLASSIC ONE)
-- ---------------------------------------------------------------------
-- For each cohort (= day of first_visit), calculate what % returned 
-- on Day 1, Day 2, ..., Day 7 after their first visit.
--
-- HOW IT WORKS:
-- 1. Find each user's cohort_date (day of first_visit)
-- 2. Find every day each user was active
-- 3. Calculate days_since_first_visit for each activity day
-- 4. Pivot to rows = cohort_date, columns = day N
-- ---------------------------------------------------------------------

WITH user_cohorts AS (
  -- Step 1: assign each user to a cohort (= their first_visit date)
  SELECT
    user_pseudo_id,
    PARSE_DATE('%Y%m%d', _TABLE_SUFFIX) AS cohort_date
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210131'
    AND event_name = 'first_visit'
),

user_activity AS (
  -- Step 2: get every day each user was active (any event)
  SELECT DISTINCT
    user_pseudo_id,
    PARSE_DATE('%Y%m%d', _TABLE_SUFFIX) AS activity_date
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210131'
),

cohort_activity AS (
  -- Step 3: join cohort assignment with activity, calculate day offset
  SELECT
    c.cohort_date,
    c.user_pseudo_id,
    a.activity_date,
    DATE_DIFF(a.activity_date, c.cohort_date, DAY) AS days_since_first_visit
  FROM user_cohorts c
  JOIN user_activity a USING (user_pseudo_id)
  WHERE a.activity_date >= c.cohort_date  -- ignore activity before first_visit (data quirks)
)

-- Step 4: pivot — rows = cohort_date, columns = day 0/1/3/7
SELECT
  cohort_date,
  COUNT(DISTINCT IF(days_since_first_visit = 0, user_pseudo_id, NULL)) AS day_0,
  COUNT(DISTINCT IF(days_since_first_visit = 1, user_pseudo_id, NULL)) AS day_1,
  COUNT(DISTINCT IF(days_since_first_visit = 3, user_pseudo_id, NULL)) AS day_3,
  COUNT(DISTINCT IF(days_since_first_visit = 7, user_pseudo_id, NULL)) AS day_7,
  -- Retention rates as percentages
  ROUND(COUNT(DISTINCT IF(days_since_first_visit = 1, user_pseudo_id, NULL)) * 100.0 
        / NULLIF(COUNT(DISTINCT IF(days_since_first_visit = 0, user_pseudo_id, NULL)), 0), 2) AS day_1_retention_pct,
  ROUND(COUNT(DISTINCT IF(days_since_first_visit = 3, user_pseudo_id, NULL)) * 100.0 
        / NULLIF(COUNT(DISTINCT IF(days_since_first_visit = 0, user_pseudo_id, NULL)), 0), 2) AS day_3_retention_pct,
  ROUND(COUNT(DISTINCT IF(days_since_first_visit = 7, user_pseudo_id, NULL)) * 100.0 
        / NULLIF(COUNT(DISTINCT IF(days_since_first_visit = 0, user_pseudo_id, NULL)), 0), 2) AS day_7_retention_pct
FROM cohort_activity
GROUP BY cohort_date
ORDER BY cohort_date
LIMIT 10;

-- Expected output (2021-01-01 to 2021-01-04 cohorts):
-- cohort_date | day_0 | day_1 | day_3 | day_7 | d1_pct | d3_pct | d7_pct
-- 2021-01-01  | 1916  | 56    | 17    | 6     | 2.92%  | 0.89%  | 0.31%
-- 2021-01-02  | 2296  | 71    | 22    | 3     | 3.09%  | 0.96%  | 0.13%
-- 2021-01-03  | 2254  | 61    | 20    | 8     | 2.71%  | 0.89%  | 0.35%
-- 2021-01-04  | 2620  | 99    | 22    | 16    | 3.78%  | 0.84%  | 0.61%
--
-- INSIGHT: Day 1 retention ~3% is LOW even for e-com browsing 
-- (typical benchmark 5-10%). Most users visit once and don't return. 
-- Day 7 ~0.3-0.6% indicates weak retention loop — likely root causes: 
-- impulse visits, weak remarketing, no email/notification re-engagement.
-- 
-- ANOMALY: Cohort 2021-01-04 shows 2x higher Day 7 retention (0.61% vs 
-- 0.13-0.35%). Worth investigating — could be a promo day, higher-quality 
-- acquisition source, or a product change. This is the kind of pattern 
-- you flag in a stakeholder review: "we have an outlier — let's find why".



-- =====================================================================
-- EDGE CASES TO REMEMBER:
-- =====================================================================
-- 1. _TABLE_SUFFIX as STRING: it's the literal date string ('20210101'), 
--    not a DATE. Use PARSE_DATE('%Y%m%d', _TABLE_SUFFIX) to convert.
--
-- 2. Truncation effect: cohorts late in the period have fewer days of 
--    "future activity" to retain. A 2021-01-30 cohort can't have day_7 
--    retention because we only have data through 2021-01-31. Filter 
--    cohorts whose full window fits in the date range for fair compare.
--
-- 3. first_visit reliability: this event fires on first session, but 
--    GA4 sometimes misses it (cookie issues). Alternative: use MIN(date)
--    per user_pseudo_id across the whole dataset as cohort assignment.
--
-- 4. Timezone: event_date is UTC. If your business operates in another 
--    TZ, retention windows shift. Adjust with DATETIME_TRUNC if needed.
--
-- 5. ALWAYS filter _TABLE_SUFFIX. Without it, BigQuery scans every 
--    historical table = expensive + slow. Use date partitions explicitly.
-- =====================================================================