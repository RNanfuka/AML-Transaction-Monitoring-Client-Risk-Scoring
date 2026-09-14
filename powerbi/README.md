# Power BI dashboard guide

After running `scripts/train_model.py`, import these CSVs into Power BI:

1. `reports/predictions.csv`
2. `reports/customer_risk_scores.csv`
3. `reports/threshold_curve.csv`

## Recommended dashboard pages

### 1. AML Executive Overview
Cards:
- Transactions evaluated
- Total alerts
- Alert rate
- Recall
- Precision
- False-positive rate

Visuals:
- Alerts by day
- Alerts by payment format
- Risk-score distribution
- Top high-risk accounts

### 2. Alert Quality / Threshold Optimization
Visuals:
- Recall vs threshold
- Precision vs threshold
- False-positive rate vs threshold
- Confusion matrix cards

Business question:
> How much suspicious activity can we capture without overwhelming investigators
> with false-positive alerts?

### 3. Client Risk Monitoring
Visuals:
- Risk tier distribution
- High-risk client table
- Total amount / transaction count by risk tier
- Maximum vs average client risk score

### 4. Transaction Pattern Analysis
Visuals:
- Payment formats associated with alerts
- Transaction amounts by risk / label
- Currency mismatch monitoring
- Cross-bank activity

## Story to tell in an interview

Do not focus only on model accuracy. Explain that AML detection is an operational
decision problem. A model with slightly lower recall may be preferable if it
substantially reduces false positives and therefore improves investigator capacity.

Use the actual metrics from `reports/metrics.json` in the final portfolio screenshots.
