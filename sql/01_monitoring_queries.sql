-- AML monitoring examples for DuckDB.
-- Run after: python scripts/build_duckdb.py

-- 1) Daily transaction velocity by sender
SELECT
    sender_id,
    CAST(timestamp AS DATE) AS txn_date,
    payment_currency AS currency,
    COUNT(*) AS txn_count,
    SUM(amount_paid) AS total_amount,
    AVG(amount_paid) AS avg_amount,
    MAX(amount_paid) AS max_amount
FROM transactions
GROUP BY 1, 2, 3
ORDER BY txn_count DESC;

-- 2) High-value transactions relative to the dataset distribution
WITH cutoff AS (
    SELECT payment_currency, quantile_cont(amount_paid, 0.99) AS p99_amount
    FROM transactions GROUP BY payment_currency
)
SELECT
    t.txn_id,
    t.timestamp,
    t.sender_id,
    t.receiver_id,
    t.amount_paid,
    t.payment_format
FROM transactions t
JOIN cutoff c USING (payment_currency)
WHERE t.amount_paid >= c.p99_amount
ORDER BY t.amount_paid DESC;

-- 3) Cross-bank / currency-mismatch monitoring
SELECT
    payment_format,
    COUNT(*) AS transactions,
    SUM(CASE WHEN from_bank <> to_bank THEN 1 ELSE 0 END) AS cross_bank_count,
    SUM(
        CASE WHEN payment_currency <> receiving_currency THEN 1 ELSE 0 END
    ) AS currency_mismatch_count,
    AVG(is_laundering) AS labelled_laundering_rate
FROM transactions
GROUP BY payment_format
ORDER BY transactions DESC;

-- 4) Accounts with many counterparties
SELECT
    sender_id,
    payment_currency,
    COUNT(*) AS transactions,
    COUNT(DISTINCT receiver_id) AS distinct_counterparties,
    SUM(amount_paid) AS total_amount
FROM transactions
GROUP BY sender_id, payment_currency
HAVING COUNT(*) >= 5
ORDER BY distinct_counterparties DESC, total_amount DESC
LIMIT 100;

-- 5) Daily inflow/outflow, matched by currency; incoming uses amount_received.
WITH flows AS (
    SELECT sender_id AS account_id, CAST(timestamp AS DATE) AS txn_date,
           payment_currency AS currency, 0.0 AS incoming, amount_paid AS outgoing
    FROM transactions
    UNION ALL
    SELECT receiver_id, CAST(timestamp AS DATE), receiving_currency, amount_received, 0.0
    FROM transactions
)
SELECT account_id, txn_date, currency, SUM(incoming) AS inflow,
       SUM(outgoing) AS outflow,
       SUM(outgoing) / NULLIF(SUM(incoming), 0) AS outflow_to_inflow_ratio
FROM flows GROUP BY account_id, txn_date, currency
ORDER BY outflow_to_inflow_ratio DESC NULLS LAST;

-- 6) High-frequency senders and account summaries (amounts separated by currency).
SELECT sender_id, payment_currency, COUNT(*) AS transactions,
       COUNT(DISTINCT receiver_id) AS counterparties,
       SUM(amount_paid) AS total_amount, MAX(amount_paid) AS maximum_amount,
       MIN(timestamp) AS first_activity, MAX(timestamp) AS last_activity
FROM transactions GROUP BY sender_id, payment_currency
ORDER BY transactions DESC LIMIT 100;

-- 7) Labelled concentration for retrospective analysis, never a predictive feature.
SELECT CAST(timestamp AS DATE) AS txn_date, payment_format,
       COUNT(*) AS transactions, SUM(is_laundering) AS labelled_transactions,
       AVG(is_laundering) AS labelled_rate
FROM transactions GROUP BY 1, 2 ORDER BY labelled_transactions DESC;
