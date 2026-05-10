-- =====================================================================
-- 02_funnel_analysis.sql
-- =====================================================================
-- BUSINESS QUESTION:
-- "Where do users drop off in the purchase funnel, and what is the
--  conversion rate at each step?"
--
-- WHY THIS MATTERS:
-- Funnel analysis is the #1 question in marketing analytics interviews.
-- Without it, you cannot answer: "Should we optimize the cart page or
-- the checkout flow?" — because you don't know where the bleeding is.
--
-- FUNNEL DEFINITION:
--   Step 1: view_item       (looked at a product)
--   Step 2: add_to_cart     (added to cart)
--   Step 3: begin_checkout  (started checkout)
--   Step 4: purchase        (completed purchase)
--
-- DATASET: bigquery-public-data.ga4_obfuscated_sample_ecommerce
-- =====================================================================


-- ---------------------------------------------------------------------
-- QUERY 1: Simple count-based funnel (NOT a real funnel — see warning)
-- ---------------------------------------------------------------------
-- This counts unique users at each step INDEPENDENTLY. It is the wrong
-- way to do funnel analysis but it's where most beginners start.
--
-- WHY IT'S WRONG:
-- A user who did `purchase` without `add_to_cart` (e.g., one-click buy)
-- still gets counted in purchase but not in add_to_cart. Funnel
-- assumes ORDERED progression — this query doesn't enforce that.
--
-- USE THIS ONLY FOR: quick smoke-check that events exist in the data.
-- ---------------------------------------------------------------------

SELECT
  event_name,
  COUNT(DISTINCT user_pseudo_id) AS unique_users
FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210131`
WHERE event_name IN ('view_item', 'add_to_cart', 'begin_checkout', 'purchase')
GROUP BY event_name
ORDER BY 
  CASE event_name
    WHEN 'view_item'      THEN 1
    WHEN 'add_to_cart'    THEN 2
    WHEN 'begin_checkout' THEN 3
    WHEN 'purchase'       THEN 4
  END;

-- Expected output (2021-01-31):
-- event_name      | unique_users
-- view_item       | 539
-- add_to_cart     | 98
-- begin_checkout  | 54
-- purchase        | 17


-- ---------------------------------------------------------------------
-- QUERY 2: Proper funnel with sequential progression (THE RIGHT WAY)
-- ---------------------------------------------------------------------
-- We mark each user with flags: did_view, did_cart, did_checkout, did_buy.
-- Then we count users who completed AT LEAST step N — this gives a
-- monotonically decreasing funnel (each step <= previous step).
--
-- This is what you'd present to a PM or marketing lead.
-- ---------------------------------------------------------------------

WITH user_steps AS (
  -- For each user, flag which funnel events they triggered (any time today).
  SELECT
    user_pseudo_id,
    MAX(IF(event_name = 'view_item',      1, 0)) AS did_view,
    MAX(IF(event_name = 'add_to_cart',    1, 0)) AS did_cart,
    MAX(IF(event_name = 'begin_checkout', 1, 0)) AS did_checkout,
    MAX(IF(event_name = 'purchase',       1, 0)) AS did_buy
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210131`
  WHERE event_name IN ('view_item', 'add_to_cart', 'begin_checkout', 'purchase')
  GROUP BY user_pseudo_id
),

funnel AS (
  -- Count users at each cumulative funnel stage.
  -- IMPORTANT: each step requires the PREVIOUS step too.
  SELECT
    SUM(did_view)                                                   AS step_1_view,
    SUM(IF(did_view = 1 AND did_cart = 1, 1, 0))                    AS step_2_cart,
    SUM(IF(did_view = 1 AND did_cart = 1 AND did_checkout = 1, 1, 0)) AS step_3_checkout,
    SUM(IF(did_view = 1 AND did_cart = 1 
         AND did_checkout = 1 AND did_buy = 1, 1, 0))               AS step_4_purchase
  FROM user_steps
)

SELECT
  step_1_view     AS view_item,
  step_2_cart     AS add_to_cart,
  step_3_checkout AS begin_checkout,
  step_4_purchase AS purchase,
  -- Conversion rates at each step (vs previous step)
  ROUND(step_2_cart     * 100.0 / NULLIF(step_1_view,     0), 2) AS view_to_cart_pct,
  ROUND(step_3_checkout * 100.0 / NULLIF(step_2_cart,     0), 2) AS cart_to_checkout_pct,
  ROUND(step_4_purchase * 100.0 / NULLIF(step_3_checkout, 0), 2) AS checkout_to_purchase_pct,
  -- End-to-end conversion (view → purchase)
  ROUND(step_4_purchase * 100.0 / NULLIF(step_1_view,     0), 2) AS overall_conversion_pct
FROM funnel;

-- Expected output (2021-01-31):
-- view_item | add_to_cart | begin_checkout | purchase | view_to_cart | cart_to_checkout | checkout_to_purchase | overall
-- 539       | 98          | 43             | 16       | 18.18%       | 43.88%           | 37.21%               | 2.97%
--
-- INSIGHT: Biggest drop-off is view → cart (-82%). Users browse heavily 
-- but rarely add to cart. Likely root causes: price comparison shopping, 
-- weak CTAs, lack of urgency. Cart → checkout (43.88%) and checkout → 
-- purchase (37.21%) are the second/third bottlenecks. Overall conversion
-- is ~3%, in line with industry benchmarks for e-commerce browsing days.


-- ---------------------------------------------------------------------
-- QUERY 3: Funnel by acquisition channel
-- ---------------------------------------------------------------------
-- Same funnel, but sliced by traffic source. Answers:
-- "Which channels send users that ACTUALLY buy, vs just browse?"
--
-- This is a common follow-up question on interviews after the basic funnel.
-- ---------------------------------------------------------------------

WITH user_steps AS (
  SELECT
    user_pseudo_id,
    -- Pick any traffic_source — it's first-touch and same for all events
    ANY_VALUE(traffic_source.medium) AS medium,
    MAX(IF(event_name = 'view_item',      1, 0)) AS did_view,
    MAX(IF(event_name = 'add_to_cart',    1, 0)) AS did_cart,
    MAX(IF(event_name = 'begin_checkout', 1, 0)) AS did_checkout,
    MAX(IF(event_name = 'purchase',       1, 0)) AS did_buy
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_20210131`
  GROUP BY user_pseudo_id
)

SELECT
  medium,
  SUM(did_view)                                                   AS view_item,
  SUM(IF(did_view = 1 AND did_cart = 1 
       AND did_checkout = 1 AND did_buy = 1, 1, 0))               AS purchase,
  ROUND(SUM(IF(did_view = 1 AND did_cart = 1 
             AND did_checkout = 1 AND did_buy = 1, 1, 0)) * 100.0 
        / NULLIF(SUM(did_view), 0), 2)                            AS conversion_pct
FROM user_steps
WHERE did_view = 1  -- only users who entered the funnel
GROUP BY medium
ORDER BY view_item DESC
LIMIT 10;

-- Expected output (2021-01-31):
-- medium    | view_item | purchase | conversion_pct
-- organic   | 177       | 3        | 1.69
-- (none)    | 130       | 7        | 5.38   ← highest converter
-- referral  | 90        | 1        | 1.11
-- <Other>   | 80        | 3        | 3.75
--
-- INSIGHT: Direct traffic ((none)/(direct)) converts 3x better than 
-- organic search (5.38% vs 1.69%). This is users with pre-formed 
-- purchase intent — they know the brand. Performance-only marketers 
-- miss this: brand investment drives the highest-converting channel,
-- not just paid acquisition. Counterintuitive but consistent across
-- e-commerce — direct = brand strength, organic = research mode.

-- =====================================================================
-- EDGE CASES TO REMEMBER:
-- =====================================================================
-- 1. Same-user-multiple-purchases: this query counts UNIQUE users per
--    step, not events. If a user buys twice, they count once. For
--    revenue analysis, count events instead.
--
-- 2. Cross-day funnels: this query is single-day. In real analysis,
--    a user might view today and buy tomorrow. Use _TABLE_SUFFIX to
--    query date ranges (covered in 03_cohort_retention.sql).
--
-- 3. Out-of-order events: GA4 lets you fire events in any sequence.
--    QUERY 2's MAX(IF()) flags handle this — we don't enforce strict
--    ordering, only that all required events happened.
--
-- 4. NULLIF(x, 0) prevents division-by-zero. Critical for percentage
--    calculations on small samples (e.g., a channel with 0 view_items).
--
-- 5. ANY_VALUE(traffic_source.medium): user might have multiple events
--    with same first-touch medium, ANY_VALUE picks one non-deterministically.
--    For deterministic results, use MIN() or MAX() — but for first-touch
--    they're all the same anyway.
-- =====================================================================