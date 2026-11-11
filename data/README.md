# Data

Primary portfolio dataset:

**IBM Transactions for Anti Money Laundering (AML)**  
Kaggle dataset: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`

Recommended file: `HI-Small_Trans.csv`.

The raw data is **not committed to this repository**. Download it yourself and
place it under:

```text
data/raw/HI-Small_Trans.csv
```

Or run:

```bash
python scripts/download_data.py
```

The IBM project documentation states that the transaction data are synthetic and
include a transaction-level laundering label. The data have their own licensing
terms, so this repository contains code only.

For a small pipeline test without Kaggle, run:

```bash
python scripts/generate_demo_data.py
```

Then train on:

```bash
python scripts/train_model.py --data data/raw/demo_ibm_aml.csv
```

