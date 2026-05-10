-- =====================================================================
-- 04_user_journey_attribution.sql
-- =====================================================================
-- BUSINESS QUESTION:
-- "When a user buys, which marketing channels touched them along the way,
--  and how should we attribute revenue across those touchpoints?"
--
-- WHY THIS MATTERS:
-- GA4 default = last-click attribution. This systematically OVER-credits
-- direct/branded search and UNDER-credits awareness channels (display,
-- social, video). Multi-touch attribution gives a more honest picture
-- of which channels actually drive revenue.
--
-- WHAT THIS QUERY PRODUCES:
-- For every purchase, the full touchpoint history → then 3 attribution 
-- models applied: first-touch, last-touch, linear.
--
-- KEY GA4 NUANCE:
-- source/medium are populated ONLY on the first event of each session.
-- We propagate them across the session using window functions.
-- =====================================================================


-- ---------------------------------------------------------------------
-- QUERY 1: Sessionize events with source/medium propagation
-- ---------------------------------------------------------------------
-- For each session, fill source/medium from the first event that has them.
-- Pattern: MAX(...) OVER (PARTITION BY session) — picks the non-NULL value
-- and applies it to every row in that session.
-- ---------------------------------------------------------------------

WITH events_with_session AS (
  SELECT
    user_pseudo_id,
    event_timestamp,
    event_name,
    -- Pull session_id from event_params (it's an int_value)
    (SELECT value.int_value 
     FROM UNNEST(event_params) 
     WHERE key = 'ga_session_id') AS session_id,
    -- Pull source/medium from event_params (string_value, mostly on session_start)
    (SELECT value.string_value 
     FROM UNNEST(event_params) 
     WHERE key = 'source') AS event_source,
    (SELECT value.string_value 
     FROM UNNEST(event_params) 
     WHERE key = 'medium') AS event_medium
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210131'
)

SELECT
  user_pseudo_id,
  session_id,
  event_timestamp,
  event_name,
  event_source,
  event_medium,
  -- Forward-fill: propagate non-NULL source/medium across the session
  MAX(event_source) OVER (PARTITION BY user_pseudo_id, session_id) AS session_source,
  MAX(event_medium) OVER (PARTITION BY user_pseudo_id, session_id) AS session_medium
FROM events_with_session
WHERE session_id IS NOT NULL
ORDER BY user_pseudo_id, event_timestamp
LIMIT 20;

-- Expected output: every event now has session_source/session_medium 
-- populated, even if event_source/event_medium were NULL on that row.


-- ---------------------------------------------------------------------
-- QUERY 2: Build user journeys for purchasers
-- ---------------------------------------------------------------------
-- For each user who made a purchase, list all unique sessions they had
-- BEFORE the purchase, in chronological order. Each session = one touchpoint.
--
-- This is the foundation of multi-touch attribution.
-- ---------------------------------------------------------------------

WITH events_with_session AS (
  SELECT
    user_pseudo_id,
    event_timestamp,
    event_name,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'source') AS event_source,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'medium') AS event_medium
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210131'
),

sessions AS (
  -- Collapse events into one row per session, with session-level source/medium
  SELECT
    user_pseudo_id,
    session_id,
    MIN(event_timestamp) AS session_start_ts,
    MAX(IF(event_name = 'purchase', 1, 0)) AS session_had_purchase,
    -- ANY_VALUE on the propagated session_source/medium
    ANY_VALUE(session_source) AS source,
    ANY_VALUE(session_medium) AS medium
  FROM (
    SELECT
      *,
      MAX(event_source) OVER (PARTITION BY user_pseudo_id, session_id) AS session_source,
      MAX(event_medium) OVER (PARTITION BY user_pseudo_id, session_id) AS session_medium
    FROM events_with_session
  )
  WHERE session_id IS NOT NULL
  GROUP BY user_pseudo_id, session_id
),

purchasers AS (
  -- Identify users who bought + the timestamp of their FIRST purchase
  SELECT
    user_pseudo_id,
    MIN(IF(session_had_purchase = 1, session_start_ts, NULL)) AS first_purchase_ts
  FROM sessions
  GROUP BY user_pseudo_id
  HAVING first_purchase_ts IS NOT NULL
),

journeys AS (
  -- Get all sessions for purchasers, up to and including the purchase session
  SELECT
    s.user_pseudo_id,
    s.session_id,
    s.session_start_ts,
    s.source,
    s.medium,
    s.session_had_purchase,
    -- Position in journey (1 = first touch, N = converting touch)
    ROW_NUMBER() OVER (
      PARTITION BY s.user_pseudo_id 
      ORDER BY s.session_start_ts
    ) AS touch_position,
    -- Total touches in this user's journey
    COUNT(*) OVER (PARTITION BY s.user_pseudo_id) AS total_touches
  FROM sessions s
  JOIN purchasers p USING (user_pseudo_id)
  WHERE s.session_start_ts <= p.first_purchase_ts
)

SELECT *
FROM journeys
ORDER BY user_pseudo_id, touch_position
LIMIT 30;

-- Expected output: rows showing full journey of each purchaser.
-- Some have 1 touch (impulse buy), others 5+ (long consideration).


-- ---------------------------------------------------------------------
-- QUERY 3: Apply 3 attribution models
-- ---------------------------------------------------------------------
-- For each purchaser's journey, distribute "1 conversion" across touches:
--   FIRST-TOUCH:  100% to touch 1
--   LAST-TOUCH:   100% to last touch (the converting one) — GA4 default
--   LINEAR:       1/N to each touch (equal credit)
--
-- Then aggregate by source/medium to compare which channels each model 
-- credits the most.
-- ---------------------------------------------------------------------

WITH events_with_session AS (
  SELECT
    user_pseudo_id,
    event_timestamp,
    event_name,
    (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'source') AS event_source,
    (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'medium') AS event_medium
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '20210101' AND '20210131'
),

sessions AS (
  SELECT
    user_pseudo_id,
    session_id,
    MIN(event_timestamp) AS session_start_ts,
    MAX(IF(event_name = 'purchase', 1, 0)) AS session_had_purchase,
    ANY_VALUE(session_source) AS source,
    ANY_VALUE(session_medium) AS medium
  FROM (
    SELECT
      *,
      MAX(event_source) OVER (PARTITION BY user_pseudo_id, session_id) AS session_source,
      MAX(event_medium) OVER (PARTITION BY user_pseudo_id, session_id) AS session_medium
    FROM events_with_session
  )
  WHERE session_id IS NOT NULL
  GROUP BY user_pseudo_id, session_id
),

purchasers AS (
  SELECT
    user_pseudo_id,
    MIN(IF(session_had_purchase = 1, session_start_ts, NULL)) AS first_purchase_ts
  FROM sessions
  GROUP BY user_pseudo_id
  HAVING first_purchase_ts IS NOT NULL
),

journeys AS (
  SELECT
    s.user_pseudo_id,
    s.source,
    s.medium,
    ROW_NUMBER() OVER (PARTITION BY s.user_pseudo_id ORDER BY s.session_start_ts) AS touch_position,
    COUNT(*) OVER (PARTITION BY s.user_pseudo_id) AS total_touches
  FROM sessions s
  JOIN purchasers p USING (user_pseudo_id)
  WHERE s.session_start_ts <= p.first_purchase_ts
),

-- Apply attribution credit per touchpoint
attributed AS (
  SELECT
    user_pseudo_id,
    COALESCE(medium, '(unknown)') AS medium,
    COALESCE(source, '(unknown)') AS source,
    -- First-touch: 1.0 if first touch, else 0
    IF(touch_position = 1, 1.0, 0) AS first_touch_credit,
    -- Last-touch: 1.0 if last touch, else 0
    IF(touch_position = total_touches, 1.0, 0) AS last_touch_credit,
    -- Linear: 1/N to every touch
    1.0 / total_touches AS linear_credit
  FROM journeys
)

-- Final aggregation: total credits per channel under each model
SELECT
  medium,
  source,
  ROUND(SUM(first_touch_credit), 2) AS first_touch_conversions,
  ROUND(SUM(last_touch_credit),  2) AS last_touch_conversions,
  ROUND(SUM(linear_credit),      2) AS linear_conversions,
  -- Highlight the gap: how does last-touch differ from linear?
  ROUND(SUM(last_touch_credit) - SUM(linear_credit), 2) AS last_minus_linear
FROM attributed
GROUP BY medium, source
ORDER BY linear_conversions DESC
LIMIT 20;

-- Expected output: per-channel comparison.
-- Channels with last_minus_linear > 0 = OVERVALUED by GA4 default.
-- Channels with last_minus_linear < 0 = UNDERVALUED by GA4 default.


-- =====================================================================
-- EDGE CASES TO REMEMBER:
-- =====================================================================
-- 1. NULL source/medium: GA4 only populates these on first session event.
--    Forward-fill via MAX() OVER (PARTITION BY session) is critical.
--
-- 2. Cross-day sessions: ga_session_id may persist or reset depending 
--    on GA4 config. We treat each (user, session_id) as one touchpoint.
--
-- 3. First-purchase only: this query attributes ONE purchase per user.
--    Repeat purchases are excluded to avoid double-counting acquisition 
--    credit. For LTV analysis, modify to attribute every purchase.
--
-- 4. Cookie/device limits: user_pseudo_id resets when cookies clear.
--    A user across two devices = two journeys, broken attribution.
--    This is the limit of client-side tracking — see project's 
--    cookieless tracking module for mitigation strategies.
--
-- 5. Linear ≠ Shapley: linear gives EQUAL credit, Shapley gives FAIR 
--    credit (game-theoretic, accounts for which channels CAUSE conversion 
--    when combined). Shapley implementation is in src/attribution/.
-- =====================================================================