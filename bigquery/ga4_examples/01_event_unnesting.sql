-- =====================================================================
-- 01_event_unnesting.sql
-- =====================================================================
-- BUSINESS QUESTION:
-- "How do we extract user-level page activity and acquisition source
--  from raw GA4 events?"
--
-- WHY THIS MATTERS:
-- GA4 stores all event parameters (page_location, session_id, etc.) inside
-- a nested ARRAY<STRUCT> called event_params. You CANNOT query GA4 data
-- without knowing how to UNNEST. This is the foundation for everything:
-- funnels, attribution, cohorts, LTV — all start here.
--
-- DATASET: bigquery-public-data.ga4_obfuscated_sample_ecommerce
-- DATE RANGE: events_20210131 (single day for fast iteration)
-- =====================================================================


-- ---------------------------------------------------------------------
-- PATTERN 1: Correlated subquery (one parameter at a time)
-- ---------------------------------------------------------------------
-- Use when: extracting 1-3 parameters, readability > performance.
-- How it works: for each row in main table, run a mini-query against
-- the unnested event_params array and pull back a single value.
-- ---------------------------------------------------------------------

SELECT
  event_name,
  user_pseudo_id,
  (SELECT value.string_value 
   FROM UNNEST(event_params) 
   WHERE key = 'page_location') AS page_location,
  (SELECT value.string_value 
   FROM UNNEST(event_params) 
   WHERE key = 'page_referrer') AS page_referrer,
  (SELECT value.int_value 
   FROM UNNEST(event_params) 
   WHERE key = 'engagement_time_msec') AS engagement_time_msec
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210131`
WHERE event_name = 'page_view'
LIMIT 10;

-- Expected output (sample):
-- event_name | user_pseudo_id      | page_location           | page_referrer | engagement_time_msec
-- page_view  | 1026454.4271112504  | https://shop.google...  | NULL          | NULL
-- page_view  | 1029692.9551304564  | https://shop.google...  | NULL          | 3


-- ---------------------------------------------------------------------
-- PATTERN 2: JOIN UNNEST + MAX(IF()) pivot (multiple parameters)
-- ---------------------------------------------------------------------
-- Use when: extracting 5+ parameters, performance matters.
-- How it works: UNNEST joins the array onto main rows (10 params = 10
-- rows per event), then GROUP BY collapses them back. MAX(IF()) acts
-- as a pivot — picks the right value for each parameter name.
--
-- Why faster: single pass over data instead of N correlated subqueries.
-- ---------------------------------------------------------------------

SELECT
  event_name,
  user_pseudo_id,
  event_timestamp,
  MAX(IF(ep.key = 'page_location',        ep.value.string_value, NULL)) AS page_location,
  MAX(IF(ep.key = 'page_referrer',        ep.value.string_value, NULL)) AS page_referrer,
  MAX(IF(ep.key = 'engagement_time_msec', ep.value.int_value,    NULL)) AS engagement_time_msec,
  MAX(IF(ep.key = 'ga_session_id',        ep.value.int_value,    NULL)) AS session_id
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210131`,
UNNEST(event_params) AS ep
WHERE event_name = 'page_view'
GROUP BY event_name, user_pseudo_id, event_timestamp
LIMIT 10;


-- ---------------------------------------------------------------------
-- PATTERN 3: traffic_source (STRUCT, not ARRAY — no UNNEST needed)
-- ---------------------------------------------------------------------
-- Use when: getting first-touch attribution data (where user came from
-- the FIRST time they visited the site).
--
-- Key difference from event_params:
--   - event_params is ARRAY<STRUCT> → needs UNNEST
--   - traffic_source is STRUCT → access fields directly with dot notation
--
-- BUSINESS QUESTION: Which acquisition channels drive the most users?
-- ---------------------------------------------------------------------

SELECT
  traffic_source.medium AS medium,
  traffic_source.source AS source,
  COUNT(DISTINCT user_pseudo_id) AS unique_users,
  COUNT(*) AS total_events,
  ROUND(COUNT(*) / COUNT(DISTINCT user_pseudo_id), 2) AS avg_events_per_user
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210131`
GROUP BY medium, source
ORDER BY unique_users DESC
LIMIT 20;

-- Expected output (2021-01-31):
-- medium    | source    | unique_users | total_events | avg_events_per_user
-- organic   | google    | 896          | 7899         | 8.82
-- (none)    | (direct)  | 625          | 7064         | 11.30
-- <Other>   | <Other>   | 405          | 3709         | 9.16
-- referral  | <Other>   | 261          | 2219         | 8.50
--
-- INSIGHT: Organic search dominates (~36% of unique users). Direct traffic
-- second (~25%) suggests strong brand recognition. CPC/paid channels are
-- bucketed under <Other> in this obfuscated public dataset (real datasets
-- show explicit cpc/google, paid_social/facebook, etc.).


-- =====================================================================
-- EDGE CASES TO REMEMBER:
-- =====================================================================
-- 1. value type matters: string_value vs int_value vs float_value vs
--    double_value. Wrong type = NULL silently. Check schema first.
--
-- 2. NULL session_id: some events fire before session is established.
--    Always handle NULLs in downstream sessionization queries.
--
-- 3. event_timestamp is in MICROSECONDS (not ms or seconds).
--    Convert with: TIMESTAMP_MICROS(event_timestamp).
--
-- 4. user_pseudo_id is the GA4 client ID (cookie-based). It changes
--    when user clears cookies — known cookieless tracking limitation.
--
-- 5. traffic_source is FIRST-TOUCH only (where user came from initially).
--    For SESSION-level source, use collected_traffic_source (newer GA4)
--    or extract source/medium from event_params per session. This matters
--    a lot for multi-touch attribution — last-touch ≠ first-touch.
-- =====================================================================