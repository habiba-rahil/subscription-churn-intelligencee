-- ============================================================
-- cohort_analysis.sql
-- ============================================================
-- SQL-based cohort retention analysis.
-- Demonstrates advanced SQL: CTEs, window functions, date math.
-- ============================================================


-- ── 1. Monthly Signup Cohort Sizes ─────────────────────────────────────────────
-- How many customers signed up each month?
SELECT
    STRFTIME(CAST(signup_date AS DATE), '%Y-%m') AS cohort_month,
    COUNT(*) AS cohort_size,
    SUM(churned) AS eventually_churned,
    ROUND(AVG(churned) * 100, 2) AS ultimate_churn_rate_pct
FROM read_csv_auto('data/processed/customers_clean.csv')
GROUP BY cohort_month
ORDER BY cohort_month;


-- ── 2. Cohort Retention at Month 1, 3, 6, 12 ───────────────────────────────────
-- Uses CASE WHEN to pivot retention at key time points.
-- A customer is "retained at month N" if they didn't churn within N months.
WITH cohort_base AS (
    SELECT
        customer_id,
        STRFTIME(CAST(signup_date AS DATE), '%Y-%m') AS cohort_month,
        churned,
        CAST(tenure_months AS INTEGER) AS tenure_m,
        CASE
            WHEN churned = 0 THEN 999  -- still active = very long tenure
            ELSE CAST(tenure_months AS INTEGER)
        END AS months_active
    FROM read_csv_auto('data/processed/customers_clean.csv')
)
SELECT
    cohort_month,
    COUNT(*) AS cohort_size,
    ROUND(SUM(CASE WHEN months_active >= 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS retained_month_1,
    ROUND(SUM(CASE WHEN months_active >= 3 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS retained_month_3,
    ROUND(SUM(CASE WHEN months_active >= 6 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS retained_month_6,
    ROUND(SUM(CASE WHEN months_active >= 12 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS retained_month_12
FROM cohort_base
GROUP BY cohort_month
ORDER BY cohort_month;


-- ── 3. Average Retention by Plan ────────────────────────────────────────────────
-- Which plans retain customers better?
WITH plan_retention AS (
    SELECT
        plan,
        customer_id,
        churned,
        CASE
            WHEN churned = 0 THEN 999
            ELSE CAST(tenure_months AS INTEGER)
        END AS months_active
    FROM read_csv_auto('data/processed/customers_clean.csv')
)
SELECT
    plan,
    COUNT(*) AS total_customers,
    ROUND(SUM(CASE WHEN months_active >= 3  THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_retained_3m,
    ROUND(SUM(CASE WHEN months_active >= 6  THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_retained_6m,
    ROUND(SUM(CASE WHEN months_active >= 12 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_retained_12m
FROM plan_retention
GROUP BY plan
ORDER BY pct_retained_12m DESC;


-- ── 4. Monthly Active Users by Month (Window Function) ─────────────────────────
-- Tracks cumulative active users using SUM OVER window.
WITH monthly AS (
    SELECT
        STRFTIME(CAST(signup_date AS DATE), '%Y-%m') AS month,
        COUNT(*) AS new_signups,
        SUM(churned) AS new_churns
    FROM read_csv_auto('data/processed/customers_clean.csv')
    GROUP BY month
)
SELECT
    month,
    new_signups,
    new_churns,
    SUM(new_signups) OVER (ORDER BY month) AS cumulative_signups,
    SUM(new_churns) OVER (ORDER BY month) AS cumulative_churns,
    SUM(new_signups) OVER (ORDER BY month) -
    SUM(new_churns) OVER (ORDER BY month) AS estimated_active_users
FROM monthly
ORDER BY month;
