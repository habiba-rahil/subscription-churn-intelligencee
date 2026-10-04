-- ============================================================
-- customer_analysis.sql
-- ============================================================
-- Core customer analytics queries using DuckDB.
--
-- HOW TO RUN:
--   python -c "import duckdb; ..."  (see notebooks/02_eda.ipynb)
--   or via DuckDB CLI: duckdb < customer_analysis.sql
--
-- WHY DUCKDB:
--   DuckDB is an embedded analytical database.
--   It can query CSV/Parquet files directly with SQL, without
--   needing to set up a server. Perfect for analytics projects.
-- ============================================================


-- ── 1. Total Customers ──────────────────────────────────────────────────────────
-- Simple count of all customer records.
SELECT
    COUNT(*) AS total_customers
FROM read_csv_auto('data/processed/customers_clean.csv');


-- ── 2. Active vs Churned Customer Counts ───────────────────────────────────────
-- CASE WHEN allows us to label numeric values with meaningful names.
SELECT
    CASE
        WHEN churned = 1 THEN 'Churned'
        ELSE 'Active'
    END AS status,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM read_csv_auto('data/processed/customers_clean.csv')
GROUP BY churned
ORDER BY churned;


-- ── 3. Overall Churn Rate ───────────────────────────────────────────────────────
-- AVG of a binary column (0/1) gives the proportion directly.
SELECT
    ROUND(AVG(churned) * 100, 2) AS churn_rate_pct,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers
FROM read_csv_auto('data/processed/customers_clean.csv');


-- ── 4. Churn Rate by Subscription Plan ─────────────────────────────────────────
-- GROUP BY to compare churn across different plan tiers.
SELECT
    plan,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(monthly_fee), 2) AS avg_monthly_fee
FROM read_csv_auto('data/processed/customers_clean.csv')
GROUP BY plan
ORDER BY churn_rate_pct DESC;


-- ── 5. Churn Rate by Country ────────────────────────────────────────────────────
SELECT
    country,
    COUNT(*) AS total_customers,
    SUM(churned) AS churned_customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate_pct
FROM read_csv_auto('data/processed/customers_clean.csv')
GROUP BY country
ORDER BY churn_rate_pct DESC;


-- ── 6. Average Monthly Revenue ─────────────────────────────────────────────────
-- Only count active customers for current MRR (Monthly Recurring Revenue).
SELECT
    ROUND(SUM(monthly_fee), 2) AS total_mrr,
    ROUND(AVG(monthly_fee), 2) AS avg_monthly_fee,
    COUNT(*) AS active_customers
FROM read_csv_auto('data/processed/customers_clean.csv')
WHERE churned = 0;


-- ── 7. Customer Tenure Analysis ─────────────────────────────────────────────────
-- Using CASE WHEN to bucket customers by tenure.
-- DATE_DIFF for computing days between dates.
SELECT
    CASE
        WHEN tenure_months < 3  THEN '0-3 months'
        WHEN tenure_months < 6  THEN '3-6 months'
        WHEN tenure_months < 12 THEN '6-12 months'
        WHEN tenure_months < 24 THEN '1-2 years'
        ELSE '2+ years'
    END AS tenure_bucket,
    COUNT(*) AS customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate_pct
FROM read_csv_auto('data/processed/customers_clean.csv')
GROUP BY tenure_bucket
ORDER BY MIN(tenure_months);


-- ── 8. Support Ticket Impact on Churn ──────────────────────────────────────────
-- Shows how churn rate increases with support ticket volume.
SELECT
    CASE
        WHEN support_tickets = 0 THEN '0 tickets'
        WHEN support_tickets = 1 THEN '1 ticket'
        WHEN support_tickets BETWEEN 2 AND 3 THEN '2-3 tickets'
        WHEN support_tickets BETWEEN 4 AND 5 THEN '4-5 tickets'
        ELSE '6+ tickets'
    END AS ticket_group,
    COUNT(*) AS customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate_pct
FROM read_csv_auto('data/processed/customers_clean.csv')
GROUP BY ticket_group
ORDER BY MIN(support_tickets);


-- ── 9. Payment Failure Impact ──────────────────────────────────────────────────
SELECT
    payment_failures,
    COUNT(*) AS customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate_pct
FROM read_csv_auto('data/processed/customers_clean.csv')
GROUP BY payment_failures
ORDER BY payment_failures;


-- ── 10. Monthly Customer Activity (Window Functions) ───────────────────────────
-- Uses window function to show running total of customer base.
-- STRFTIME converts date to year-month string.
WITH monthly_signups AS (
    SELECT
        STRFTIME(CAST(signup_date AS DATE), '%Y-%m') AS signup_month,
        COUNT(*) AS new_customers,
        SUM(churned) AS churned_this_cohort
    FROM read_csv_auto('data/processed/customers_clean.csv')
    GROUP BY signup_month
)
SELECT
    signup_month,
    new_customers,
    churned_this_cohort,
    SUM(new_customers) OVER (ORDER BY signup_month) AS cumulative_customers,
    ROUND(churned_this_cohort * 100.0 / new_customers, 2) AS cohort_churn_rate_pct
FROM monthly_signups
ORDER BY signup_month;
