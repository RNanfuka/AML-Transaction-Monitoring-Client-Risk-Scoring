from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def main(n_normal: int = 6000, n_laundering: int = 160, seed: int = 42):
    rng = np.random.default_rng(seed)
    out = Path("data/raw/demo_ibm_aml.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    accounts = [f"A{n:05d}" for n in range(1, 900)]
    banks = np.arange(1, 26)
    formats = np.array(["ACH", "Wire", "Cheque", "Credit Card", "Cash"])
    currencies = np.array(["USD", "CAD", "EUR", "GBP"])

    def rows(n, laundering):
        start = pd.Timestamp("2025-01-01")
        minutes = rng.integers(0, 60 * 24 * 90, n)
        ts = start + pd.to_timedelta(minutes, unit="m")
        amount = (
            rng.lognormal(mean=7.0, sigma=1.0, size=n)
            if not laundering
            else rng.lognormal(mean=9.0, sigma=1.2, size=n)
        )
        payment_format = (
            rng.choice(formats, n, p=[0.35, 0.18, 0.10, 0.30, 0.07])
            if not laundering
            else rng.choice(formats, n, p=[0.08, 0.62, 0.03, 0.07, 0.20])
        )
        pay_cur = rng.choice(currencies, n, p=[0.55, 0.25, 0.12, 0.08])
        recv_cur = pay_cur.copy()
        if laundering:
            swap = rng.random(n) < 0.35
            recv_cur[swap] = rng.choice(currencies, swap.sum())
        return pd.DataFrame(
            {
                "Timestamp": ts.strftime("%Y/%m/%d %H:%M"),
                "From Bank": rng.choice(banks, n),
                "Account": rng.choice(accounts, n),
                "To Bank": rng.choice(banks, n),
                "Account.1": rng.choice(accounts, n),
                "Amount Received": amount * rng.uniform(0.98, 1.02, n),
                "Receiving Currency": recv_cur,
                "Amount Paid": amount,
                "Payment Currency": pay_cur,
                "Payment Format": payment_format,
                "Is Laundering": int(laundering),
            }
        )

    df = pd.concat([rows(n_normal, False), rows(n_laundering, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} demo transactions to {out}")
    print("Use this only to test the pipeline; use IBM HI-Small for portfolio results.")


if __name__ == "__main__":
    main()
