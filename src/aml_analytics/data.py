from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required IBM AML columns: "
            + ", ".join(missing)
        )
    return df


def load_ibm_transactions(
    path: str | Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load and normalize the IBM Transactions for AML CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, nrows=nrows, dtype=str)
    df = _clean_columns(df)

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    out["from_bank"] = df["From Bank"].astype(str)
    out["sender_account"] = df["Account"].astype(str)
    out["to_bank"] = df["To Bank"].astype(str)
    out["receiver_account"] = df["Account.1"].astype(str)

    out["amount_received"] = pd.to_numeric(df["Amount Received"], errors="coerce")
    out["receiving_currency"] = df["Receiving Currency"].astype(str)
    out["amount_paid"] = pd.to_numeric(df["Amount Paid"], errors="coerce")
    out["payment_currency"] = df["Payment Currency"].astype(str)
    out["payment_format"] = df["Payment Format"].astype(str)
    labels = pd.to_numeric(df["Is Laundering"], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("Is Laundering must contain only 0 or 1; missing labels are invalid.")
    out["is_laundering"] = labels.astype(int)
    identifiers = ["From Bank", "Account", "To Bank", "Account.1",
                   "Payment Currency", "Receiving Currency", "Payment Format"]
    if df[identifiers].isna().any().any() or df[identifiers].apply(lambda c: c.str.strip().eq("")).any().any():
        raise ValueError("Missing account, bank, currency or payment-format values.")
    amounts = out[["amount_paid", "amount_received"]]
    if out.timestamp.isna().any() or not np.isfinite(amounts).all().all() or (amounts < 0).any().any():
        raise ValueError("Invalid timestamp or non-finite/negative transaction amount.")

    out["sender_id"] = out["from_bank"] + "::" + out["sender_account"]
    out["receiver_id"] = out["to_bank"] + "::" + out["receiver_account"]
    out["txn_id"] = [f"T{i:09d}" for i in range(1, len(out) + 1)]

    out = out.dropna(subset=["timestamp", "amount_paid"])
    out = out[out["amount_paid"] >= 0].reset_index(drop=True)
    return out


def temporal_split(
    df: pd.DataFrame,
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be between 0.5 and 1.0")
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    if len(ordered) < 2:
        raise ValueError("At least two rows are required.")
    boundary = ordered.iloc[int(len(ordered) * train_fraction)]["timestamp"]
    train, test = ordered[ordered.timestamp < boundary], ordered[ordered.timestamp >= boundary]
    if train.empty or test.empty:
        raise ValueError("Insufficient distinct timestamps for chronological splitting.")
    return train.copy(), test.copy()


def stratified_downsample(
    df: pd.DataFrame,
    max_rows: int | None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Keep every laundering row and sample non-laundering rows up to max_rows.
    Useful for laptop-friendly experimentation on very large files.
    """
    if max_rows is None or len(df) <= max_rows:
        return df.copy()

    positives = df[df["is_laundering"] == 1]
    negatives = df[df["is_laundering"] == 0]

    remaining = max(max_rows - len(positives), 0)
    if remaining == 0:
        return positives.sample(
            n=min(max_rows, len(positives)),
            random_state=random_state
        ).copy()

    sampled_neg = negatives.sample(
        n=min(remaining, len(negatives)),
        random_state=random_state,
    )
    return (
        pd.concat([positives, sampled_neg], ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
