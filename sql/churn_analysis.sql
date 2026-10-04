-- ============================================================
-- churn_analysis.sql
-- ============================================================
-- Deep-dive churn analysis queries.
-- Demonstrates: CTEs, window functions, subqueries, CASE WHEN.
-- ============================================================


-- ── 1. High-Risk Customer Identification ───────────────────────────────────────
-- Finds active customers with multiple risk signals.
-- CTE (Common Table Expression) makes the query readable.
WITH risk_signals AS (
    SELECT
        customer_id,
        plan,
        tenure_months,
        usage_frequency,
        payment_failures,
        support_tickets,
        days_since_login,
        -- Build a simple risk score
        (CASE WHEN usage_frequency < 5 THEN 2 ELSE 0 END) +
        (CASE WHEN payment_failures >= 2 THEN 3 ELSE 0 END) +
        (CASE WHEN support_tickets >= 3 THEN 2 ELSE 0 END) +
        (CASE WHEN days_since_login > 30 THEN 1 ELSE 0 END) AS risk_score
    FROM read_csv_auto('data/processed/customers_clean.csv')
    WHERE churned = 0  -- Only active customers
)
SELECT
    customer_id,
    plan,
    tenure_months,
    usage_frequency,
    payment_failures,
    support_tickets,
    days_since_login,
    risk_score,
    CASE
        WHEN risk_score >= 5 THEN 'High Risk'
        WHEN risk_score >= 3 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS risk_category
FROM risk_signals
ORDER BY risk_score DESC
LIMIT 50;


-- ── 2. Churn Segmentation by Multiple Dimensions ────────────────────────────────
-- Shows churn rate for plan + payment method combinations.
SELECT
    plan,
    payment_method,
    COUNT(*) AS customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate_pct
FROM read_csv_auto('data/processed/customers_clean.csv')
GROUP BY plan, payment_method
HAVING COUNT(*) >= 20  -- Only show meaningful segments
ORDER BY churn_rate_pct DESC;


-- ── 3. Revenue at Risk ─────────────────────────────────────────────────────────
-- How much recurring revenue is at risk from high-risk customers?
WITH risk_customers AS (
    SELECT
        customer_id,
        monthly_fee,
        (CASE WHEN usage_frequency < 5 THEN 2 ELSE 0 END) +
        (CASE WHEN payment_failures >= 2 THEN 3 ELSE 0 END) +
        (CASE WHEN support_tickets >= 3 THEN 2 ELSE 0 END) AS risk_score
    FROM read_csv_auto('data/processed/customers_clean.csv')
    WHERE churned = 0
)
SELECT
    CASE
        WHEN risk_score >= 5 THEN 'High Risk'
        WHEN risk_score >= 3 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS risk_category,
    COUNT(*) AS customer_count,
    ROUND(SUM(monthly_fee), 2) AS monthly_revenue_at_risk,
    ROUND(AVG(monthly_fee), 2) AS avg_monthly_fee
FROM risk_customers
GROUP BY risk_category
ORDER BY monthly_revenue_at_risk DESC;


-- ── 4. Churn by Payment Failure + Usage (Cross-tab) ────────────────────────────
-- Demonstrates how churn risk compounds when multiple signals align.
SELECT
    CASE WHEN payment_failures = 0 THEN 'No Failures'
         WHEN payment_failures = 1 THEN '1 Failure'
         ELSE '2+ Failures'
    END AS payment_risk,
    CASE WHEN usage_frequency < 5 THEN 'Low Usage'
         WHEN usage_frequency < 15 THEN 'Medium Usage'
         ELSE 'High Usage'
    END AS usage_segment,
    COUNT(*) AS customers,
    ROUND(AVG(churned) * 100, 2) AS churn_rate_pct
FROM read_csv_auto('data/processed/customers_clean.csv')
GROUP BY payment_risk, usage_segment
HAVING COUNT(*) >= 30
ORDER BY churn_rate_pct DESC;


-- ── 5. Top Churned Customer Profiles ──────────────────────────────────────────
-- Shows the most common profiles among churned customers.
SELECT
    plan,
    country,
    payment_method,
    CASE WHEN age < 35 THEN 'Young (<35)'
         WHEN age < 55 THEN 'Middle (35-54)'
         ELSE 'Senior (55+)'
    END AS age_group,
    COUNT(*) AS churned_count,
    ROUND(AVG(tenure_months), 1) AS avg_tenure_months,
    ROUND(AVG(usage_frequency), 1) AS avg_usage
FROM read_csv_auto('data/processed/customers_clean.csv')
WHERE churned = 1
GROUP BY plan, country, payment_method, age_group
ORDER BY churned_count DESC
LIMIT 20;
