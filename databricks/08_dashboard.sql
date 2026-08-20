-- ============================================================================
-- 08 — Dashboard views (Phase 9)
--
-- Run once in the SQL Editor. Creates the views the AI/BI dashboard reads.
--
-- Why views rather than queries pasted into dashboard widgets:
--   * the logic is versioned in git and reviewable
--   * every widget agrees on definitions - "high risk" means one thing
--   * Free Edition gives one 2X-Small warehouse, so widgets hit small
--     pre-aggregated views instead of scanning bronze
--
-- Change the two lines below if you used a different catalog/schema.
-- ============================================================================

USE CATALOG workspace;
USE SCHEMA payment_ops;


-- ---------------------------------------------------------------------------
-- 1. Executive KPIs — one row, the numbers a CFO would ask for first
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_executive_kpis AS
SELECT
    COUNT(*)                                                     AS open_invoices,
    ROUND(SUM(invoice_amount), 2)                                AS total_outstanding,
    SUM(CASE WHEN days_late_pred > 0 THEN 1 ELSE 0 END)          AS predicted_late,
    ROUND(SUM(CASE WHEN days_late_pred > 0 THEN invoice_amount ELSE 0 END), 2)
                                                                 AS value_at_risk,
    SUM(CASE WHEN action = 'CALL' THEN 1 ELSE 0 END)             AS recommended_calls,
    ROUND(SUM(CASE WHEN action = 'CALL' THEN expected_value ELSE 0 END), 2)
                                                                 AS expected_daily_recovery,
    ROUND(AVG(days_late_pred), 2)                                AS avg_predicted_delay,
    ROUND(AVG(days_late_upper - days_late_lower), 2)             AS avg_interval_width,
    MAX(queue_date)                                              AS queue_generated_at
FROM gold_collection_queue;


-- ---------------------------------------------------------------------------
-- 2. Today's collection queue — the work order
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_collection_queue AS
SELECT
    `rank`,
    action,
    customer_name,
    invoice_id,
    ROUND(invoice_amount, 2)        AS amount,
    due_date,
    predicted_payment_date,
    CAST(CEIL(days_late_pred) AS INT) AS days_late_predicted,
    CONCAT(CAST(CEIL(days_late_lower) AS INT), ' to ',
           CAST(CEIL(days_late_upper) AS INT), ' days') AS prediction_interval,
    ROUND(p_late, 2)                AS p_late,
    ROUND(p_responds, 2)            AS p_responds,
    ROUND(expected_value, 2)        AS expected_value
FROM gold_collection_queue
ORDER BY `rank`;


-- ---------------------------------------------------------------------------
-- 3. Action tiers — where the exposure sits
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_action_summary AS
SELECT
    action,
    COUNT(*)                                AS invoices,
    ROUND(SUM(invoice_amount), 2)           AS exposure,
    ROUND(SUM(expected_value), 2)           AS total_expected_value,
    ROUND(AVG(p_late), 3)                   AS avg_p_late,
    ROUND(AVG(days_late_pred), 2)           AS avg_predicted_delay
FROM gold_collection_queue
GROUP BY action;


-- ---------------------------------------------------------------------------
-- 4. Invoice drill-down — everything known about one invoice
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_invoice_detail AS
SELECT
    q.invoice_id,
    q.customer_id,
    q.customer_name,
    ROUND(q.invoice_amount, 2)              AS amount,
    q.posting_date,
    q.due_date,
    q.predicted_payment_date,
    ROUND(q.days_late_pred, 1)              AS p50_days_late,
    ROUND(q.days_late_lower, 1)             AS p10_days_late,
    ROUND(q.days_late_upper, 1)             AS p90_days_late,
    ROUND(q.p_late, 3)                      AS probability_late,
    ROUND(q.expected_value, 2)              AS expected_value,
    q.action,
    q.payment_terms,
    q.business_code,
    -- Customer context: why this invoice looks the way it does.
    CAST(q.cust_invoice_count AS INT)       AS customer_history_depth,
    ROUND(c.avg_days_late, 1)               AS customer_avg_days_late,
    ROUND(c.late_rate, 3)                   AS customer_late_rate,
    ROUND(c.recent_90d_avg_days_late, 1)    AS customer_recent_90d_avg,
    q.model_version,
    q.scored_at
FROM gold_collection_queue q
LEFT JOIN gold_customer_payment_behavior c
       ON q.customer_id = c.customer_id;


-- ---------------------------------------------------------------------------
-- 5. Customer risk segmentation — median split into four quadrants
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_customer_segments AS
WITH medians AS (
    SELECT percentile_approx(avg_days_late, 0.5) AS med_late,
           percentile_approx(std_days_late, 0.5) AS med_std
    FROM gold_customer_payment_behavior
    WHERE std_days_late IS NOT NULL
)
SELECT
    c.customer_id,
    ROUND(c.avg_days_late, 1)       AS avg_days_late,
    ROUND(c.std_days_late, 1)       AS payment_variability,
    c.invoice_count,
    ROUND(c.avg_invoice_amount, 2)  AS avg_invoice_amount,
    CASE
        WHEN c.avg_days_late >  m.med_late AND c.std_days_late >  m.med_std THEN 'Late & Erratic'
        WHEN c.avg_days_late >  m.med_late AND c.std_days_late <= m.med_std THEN 'Late but Predictable'
        WHEN c.avg_days_late <= m.med_late AND c.std_days_late >  m.med_std THEN 'Prompt but Erratic'
        ELSE 'Prompt & Reliable'
    END AS segment
FROM gold_customer_payment_behavior c
CROSS JOIN medians m
WHERE c.std_days_late IS NOT NULL;


-- ---------------------------------------------------------------------------
-- 6. Model performance on delayed labels — trend, not snapshot
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_model_performance AS
SELECT
    evaluated_at,
    model_version,
    ROUND(mae, 3)                   AS mae,
    ROUND(rmse, 3)                  AS rmse,
    ROUND(bias, 3)                  AS bias,
    ROUND(within_3d, 1)             AS pct_within_3_days,
    ROUND(pi_coverage, 1)           AS pi_coverage_pct,
    ROUND(late_auc, 3)              AS late_auc,
    n_labelled,
    n_awaiting_label,
    median_label_lag_days
FROM gold_model_performance
ORDER BY evaluated_at DESC;


-- ---------------------------------------------------------------------------
-- 7. Drift — latest PSI per feature
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_drift_latest AS
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY feature ORDER BY evaluated_at DESC) AS rn
    FROM gold_drift_metrics
)
SELECT feature, `type`, ROUND(psi, 4) AS psi, status, model_version, evaluated_at
FROM ranked
WHERE rn = 1
ORDER BY psi DESC;


-- ---------------------------------------------------------------------------
-- 8. Pipeline health — data quality across runs
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_pipeline_health AS
SELECT
    run_timestamp,
    stage,
    rows_in,
    rows_clean,
    rows_quarantined,
    ROUND(reject_rate * 100, 2) AS reject_rate_pct,
    reject_reasons
FROM silver_data_quality_metrics
ORDER BY run_timestamp DESC;


CREATE OR REPLACE VIEW v_quarantine_reasons AS
SELECT _reject_reason AS reason, COUNT(*) AS rows_rejected
FROM silver_quarantine
GROUP BY _reject_reason
ORDER BY rows_rejected DESC;


-- ---------------------------------------------------------------------------
-- 9. Strategy comparison — is the ranking worth its complexity?
--
-- Compares the EV captured by the engine's top 20 against the two strategies a
-- collections team would otherwise use. Self-consistent under the engine's own
-- assumptions: it compares ORDERINGS, and is not a measured business outcome.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_strategy_comparison AS
WITH by_ev AS (
    SELECT SUM(expected_value) AS ev FROM (
        SELECT expected_value FROM gold_collection_queue ORDER BY expected_value DESC LIMIT 20) t),
by_amount AS (
    SELECT SUM(expected_value) AS ev FROM (
        SELECT expected_value FROM gold_collection_queue ORDER BY invoice_amount DESC LIMIT 20) t),
at_random AS (
    SELECT SUM(expected_value) AS ev FROM (
        SELECT expected_value FROM gold_collection_queue ORDER BY RAND(42) LIMIT 20) t)
SELECT 'Decision engine' AS strategy, ROUND(ev, 2) AS expected_value_of_20_calls, 1 AS sort_order FROM by_ev
UNION ALL
SELECT 'By amount',        ROUND(ev, 2), 2 FROM by_amount
UNION ALL
SELECT 'Random',           ROUND(ev, 2), 3 FROM at_random
ORDER BY sort_order;


-- ---------------------------------------------------------------------------
-- Verify
-- ---------------------------------------------------------------------------
SELECT * FROM v_executive_kpis;
