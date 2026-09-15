# AML Transaction Monitoring & Client Risk Scoring

An AML analytics portfolio project exploring how transaction monitoring teams can
detect labelled suspicious activity while managing investigator alert volume.
Python models, SQL analysis and a Streamlit dashboard connect transaction patterns
to account prioritization and measurable false-positive trade-offs.

**Current status: IBM HI-Small training and held-out evaluation completed.
Results below use the IBM synthetic dataset with explicitly disclosed row caps.**

## Business problem and objectives

Investigators have limited review capacity. This project measures detection recall
alongside precision, false positives and alert volume, compares interpretable and
non-linear models, and presents account activity in a reviewable dashboard.
A model flag is a prompt for investigation, not evidence of criminal activity.

## Dataset

The intended source is IBM Transactions for Anti Money Laundering, specifically
`HI-Small_Trans.csv`. It is synthetic, transaction-labelled data.
Download the dataset named
`ealtman2019/ibm-transactions-for-anti-money-laundering-aml` from Kaggle and place
the file at **`data/raw/HI-Small_Trans.csv`**.

With Kaggle access configured, run `python scripts/download_data.py`. The helper
requests only HI-Small, copies it from the Kaggle cache and preserves an existing
local file. Authentication and network access are required for a live download;
the download adapter is tested with a mocked cache.

The loader checks required columns, binary labels, timestamps, finite nonnegative
amounts, and missing identifiers/categories. Account identifiers retain leading
zeros. `reports/dataset_summary.json` records row count, labelled count and rate,
date range, and unique sender/receiver counts. Raw datasets are ignored by Git.

## Architecture and technologies

```text
IBM CSV → schema validation → chronological 60/20/20 periods
        → strictly historical features → train three models
        → validation model + threshold selection → held-out test evaluation
        → reports / saved model → SQL analysis + Streamlit dashboard
```

Python 3.11, pandas, NumPy, scikit-learn, DuckDB, Plotly, Streamlit, joblib,
pytest and GitHub Actions. Dependencies are declared in `pyproject.toml`;
`requirements.txt` also includes pytest.

## Feature engineering

Features include amount/log amount, hour, weekday/weekend, cross-bank and currency
mismatch indicators, payment format and currencies. Behavioural features include:

- Prior sender count, mean, standard deviation and maximum amount.
- Prior receiver count, mean and standard deviation.
- Amount relative to sender mean and amount z-score.
- Sender transaction count over the preceding 24 hours.
- Historical unique counterparties, repeated pair count and pair concentration.
- Historical hour frequency and incoming transaction count for the sender.

All histories exclude the current transaction, later transactions and other
transactions with the same timestamp. Amount statistics and activity histories
are currency-specific. Receiver amounts use received currency and received value.
Cold-start histories are zero; ratios use a denominator floor of one.

Validation and test replay earlier unlabelled activity in time order, including
validation history before test observations. Labels are never feature inputs.
The saved engineer contains training history; replay intervening transactions
when scoring a later period to reproduce this evaluation.

## Modelling approach

Logistic Regression provides an interpretable baseline; Random Forest and
HistGradientBoosting provide non-linear comparisons without new dependencies.
Preprocessing and models fit only training rows. Gradient boosting random internal
early stopping is disabled to preserve the chronological design.

Timestamp boundaries keep equal-time transactions together. Approximate periods
are 60% training, 20% validation and 20% test. Row caps uniformly sample each period
independently of labels. Feature construction updates history from the complete
input while retaining only sampled output rows to reduce memory usage.
The loader still reads the entire CSV into memory; row caps limit model workloads,
not ingestion memory. All three periods must contain both labels.

Models are ranked by validation PR-AUC, with alert volume as a tie-breaker.
The test set measures the selected model once and does not select its threshold.
`model_comparison.csv` contains validation PR-AUC, ROC-AUC, precision, recall,
F1, false-positive rate, confusion counts and alert volume.

**PR-AUC here means average precision**, not trapezoidal integration. It reflects
precision across recall levels and is useful when positives are rare. Recall
measures how many labelled transactions are detected; precision measures the
share of alerts carrying positive labels. Neither accuracy nor ROC-AUC alone
describes investigator workload.

## False-positive optimization

The validation threshold table evaluates every distinct model score plus a
no-alert boundary. The chosen threshold minimizes alert count while attaining
at least 80% validation recall by default. Tied scores are treated together.
`reports/threshold_curve.csv` includes threshold, precision, recall, F1,
false-positive rate, all four confusion counts, number of alerts and alert rate.

The recall target is **not a guarantee on future data**. Report held-out recall
and false positives even when they miss the target; do not retune on the test set.

## Client risk scoring

Each sending account receives transaction count, total amount (blank for mixed
currencies), maximum/mean score, flagged count/proportion, counterparty count and
labelled count. Labels are descriptive and do not set tiers.

Tier cutoffs are the median and 90th percentile of validation account maximum
scores. Test accounts above the 90th percentile are High; above the median are
Medium; all others are Low. Strict comparisons keep ties together and can leave
tiers empty. Cutoffs are saved in metrics.json. These are relative review
priorities, not calibrated probabilities or regulatory risk classifications.

## Explainability

`reports/feature_importance.csv` exports signed Logistic Regression coefficients
and Random Forest impurity importance. Positive coefficients increase model
scores; negative ones reduce scores, conditional on other features. Numeric
coefficients use standardized inputs; categorical coefficients use encoded levels.
Unusual amounts, concentrated counterparties and high frequency may influence
scores, but associations do not establish causality. Forest importance has
cardinality bias. HistGradientBoosting has no native importance export here.

## SQL analytics

`sql/01_monitoring_queries.sql` contains daily velocity, currency-specific
high-value cutoffs, cross-bank/currency mismatches, counterparty diversity,
currency-matched inflow/outflow ratios, account summaries and labelled
concentrations. Labelled concentration is retrospective analysis only.

```bash
python scripts/build_duckdb.py --data data/raw/HI-Small_Trans.csv
# If the DuckDB CLI is installed:
duckdb data/processed/aml.duckdb < sql/01_monitoring_queries.sql
```

## Dashboard

Executive KPIs cover transactions, alerts, alert rate, recall, precision,
false-positive rate and PR-AUC. Sections include score distribution, alert trends,
payment formats, highest-risk accounts, recent alerts, threshold trade-offs,
model comparisons and feature importance.

Date, payment-format and account-tier filters update test KPIs and activity.
Validation comparison and threshold tables remain explicitly labelled as full
validation results. Account tiers use fixed validation cutoffs.

## Project structure

```text
app/                 Streamlit dashboard
src/aml_analytics/   Loading, historical features, models and account scoring
scripts/             Demo generation, download, training and DuckDB creation
sql/                 Monitoring queries
tests/               Leakage, validation, thresholds, scoring and integration checks
notebooks/           Exploratory notebook
powerbi/             Optional Power BI guide
data/raw/            Local ignored CSV inputs
data/processed/      Local ignored database outputs
models/              Local ignored fitted model
reports/             Local ignored metrics and CSV reports
.github/workflows/   Python 3.11 CI
```

## Setup and execution

Use the existing Conda environment:

```bash
conda activate aml-analytics
python --version
python -m pip install -e ".[dev]"
pytest -q
```

Run the demo (generate only if missing; generation replaces the demo CSV):

```bash
python scripts/generate_demo_data.py
python scripts/train_model.py --data data/raw/demo_ibm_aml.csv --max-train-rows 5000 --max-test-rows 2000 --target-recall 0.80
```

Run IBM HI-Small after downloading:

```bash
python scripts/download_data.py
python scripts/train_model.py --data data/raw/HI-Small_Trans.csv --max-train-rows 150000 --max-test-rows 60000 --target-recall 0.80
streamlit run app/streamlit_app.py
```

Each training run replaces local reports and the model bundle. The dashboard
identifies its input source. A filename identifies IBM input but does not
authenticate dataset provenance; download from the intended source.

## Public dashboard deployment

The app supports a compact, static snapshot in `app/dashboard_data/` when local
`reports/` files are absent. Prepare it after verifying an IBM training run:

```bash
python scripts/prepare_dashboard_data.py
```

The snapshot contains compressed derived test predictions (dashboard columns only),
account scores, validation thresholds, metrics and importance. It does not include
the 454 MB raw CSV or a fitted model. All reported test metrics retain the complete
60,000-row evaluation sample. Local training reports take precedence when present.

After committing and pushing the snapshot and app code, sign into Streamlit
Community Cloud at https://share.streamlit.io/ and create an app with:

- Repository: `RNanfuka/AML-Transaction-Monitoring-Client-Risk-Scoring`
- Branch: `main`
- Entrypoint: `app/streamlit_app.py`
- Python: `3.11`

The root `requirements.txt` installs the local package with `-e .`.
Choose an available app subdomain and use the URL issued after successful
deployment on your résumé. No public deployment URL has been verified yet.
The published app must be publicly accessible for recruiters to open it.

## Verified results — IBM HI-Small (synthetic)

Verified locally on September 14, 2026 with Python 3.11 in `aml-analytics`.
The complete input contains **5,078,345 transactions**, **5,177 laundering-labelled
rows (0.101943%)**, **496,999 unique senders** and **420,640 unique receivers**.
Dates span September 1, 2022 00:00 to September 18, 2022 16:18.

All rows contribute to available chronological history. Uniform, label-independent
samples contain 150,000 training rows, 60,000 validation rows and 60,000 test rows.
The test sample has 106 positives (0.176667%); these are sampled test results,
not metrics over all 5.08 million transactions.

| Held-out IBM result | Value |
|---|---:|
| Selected model | HistGradientBoostingClassifier |
| Operating threshold | 0.026684940512 |
| PR-AUC (average precision) | 0.229403 |
| ROC-AUC | 0.972302 |
| Precision | 3.9895% |
| Recall | 85.8491% |
| F1 | 0.076246 |
| False-positive rate | 3.6565% |
| Alerts / evaluated transactions | 2,281 / 60,000 |
| Alert rate | 3.8017% |
| TP / FP / FN / TN | 91 / 2,190 / 15 / 57,704 |

The operating threshold achieved 80.2817% recall on validation and was frozen
before test evaluation. On test it detected 91 of 106 labelled transactions.
However, 2,190 of 2,281 alerts were false positives: approximately 96% of alerts
require review without a positive dataset label. This remains a substantial
investigation workload.

| Model | Validation PR-AUC | Validation alerts at target recall |
|---|---:|---:|
| HistGradientBoosting | 0.243791 | 1,986 |
| Random Forest | 0.194048 | 1,300 |
| Logistic Regression | 0.048361 | 1,522 |

The automatic selection prioritizes PR-AUC. Random Forest produced fewer
validation alerts at the same achieved recall and is worth considering when
review capacity is the primary constraint. This comparison does not retune
selection using test labels.

Metrics were independently recomputed from exported predictions, and the saved
model reproduced scores exactly when numeric CSV features were loaded with
`float_precision="round_trip"`. Generated reports and models remain ignored by
Git. Demo smoke checks run separately and do not replace these IBM results.

## Limitations and future improvements

Synthetic patterns do not establish real-world effectiveness. Small positive
counts make metrics unstable. Uniform row caps introduce sampling variation.
Scores are uncalibrated; quantile account tiers and maximum scores can favor
high-activity accounts. No currency conversion or external client/KYC context is
available. Features require complete available history and can consume substantial
memory on IBM data.

Future work: larger temporal evaluations, confidence intervals, investigator
capacity targets, probability calibration, permutation importance, rolling
backtests, streaming state and drift monitoring.

## Responsible use

This portfolio project uses synthetic data and is not a production AML decisioning system. Model alerts indicate transactions for further review, not proof of money laundering.

## Author

Rebecca Rosette Nanfuka — Master of Data Science, University of British Columbia
