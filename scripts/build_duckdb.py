from __future__ import annotations

import argparse
from pathlib import Path
import duckdb

from aml_analytics.data import load_ibm_transactions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/HI-Small_Trans.csv")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--db", default="data/processed/aml.duckdb")
    args = parser.parse_args()

    df = load_ibm_transactions(args.data, nrows=args.nrows)
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.db)
    con.register("txn_df", df)
    con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM txn_df")
    con.execute("CREATE INDEX IF NOT EXISTS idx_sender ON transactions(sender_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON transactions(timestamp)")
    print(con.execute("SELECT COUNT(*) AS rows FROM transactions").fetchdf())
    con.close()
    print(f"Created {args.db}")


if __name__ == "__main__":
    main()
