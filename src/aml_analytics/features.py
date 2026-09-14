from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import math
import numpy as np
import pandas as pd


def _account():
    return {"n": 0, "mean": 0., "m2": 0., "max": 0., "peers": {}, "hours": [0]*24, "recent": deque()}


class AMLFeatureEngineer:
    """Online features: score timestamp batches before updating unlabelled history.

    Amount history is currency-specific. transform copies fitted state so repeated
    inference is deterministic; it never updates the persisted training state.
    """
    categorical_features = ["payment_format", "payment_currency", "receiving_currency"]
    numeric_features = [
        "amount_paid", "amount_log", "hour", "day_of_week", "is_weekend",
        "cross_bank", "currency_mismatch", "sender_txn_count_hist",
        "sender_amount_mean_hist", "sender_amount_std_hist", "sender_amount_max_hist",
        "receiver_txn_count_hist", "receiver_amount_mean_hist", "receiver_amount_std_hist",
        "amount_to_sender_mean", "sender_amount_zscore", "sender_velocity_24h",
        "sender_unique_counterparties", "repeated_pair_count", "counterparty_concentration",
        "unusual_hour", "sender_incoming_count_hist",
    ]

    def fit(self, df):
        self.fit_transform(df)
        return self

    def fit_transform(self, df):
        self.state_ = ({}, {})
        out = self._run(df, self.state_)
        self.last_timestamp_ = df.timestamp.max()
        return out

    def transform(self, df):
        if not hasattr(self, "state_"):
            raise RuntimeError("Call fit() before transform().")
        if len(df) and df.timestamp.min() <= self.last_timestamp_:
            raise ValueError("Inference timestamps must be strictly after fitted history.")
        return self._run(df, deepcopy(self.state_))

    def _run(self, df, state):
        senders, receivers = state
        records = []
        for timestamp, batch in df.sort_values("timestamp", kind="stable").groupby("timestamp", sort=False):
            for r in batch.to_dict("records"):
                s = senders.setdefault((r["sender_id"], r["payment_currency"]), _account())
                v = receivers.setdefault((r["receiver_id"], r["receiving_currency"]), _account())
                while s["recent"] and s["recent"][0] < timestamp - pd.Timedelta(hours=24):
                    s["recent"].popleft()
                std = math.sqrt(s["m2"] / (s["n"]-1)) if s["n"] > 1 else 0.
                rstd = math.sqrt(v["m2"] / (v["n"]-1)) if v["n"] > 1 else 0.
                pair = s["peers"].get(r["receiver_id"], 0)
                amount = r["amount_paid"]
                r.update(dict(zip(self.numeric_features, [
                    amount, np.log1p(amount), timestamp.hour, timestamp.dayofweek,
                    int(timestamp.dayofweek >= 5), int(r["from_bank"] != r["to_bank"]),
                    int(r["payment_currency"] != r["receiving_currency"]),
                    s["n"], s["mean"], std, s["max"], v["n"], v["mean"], rstd,
                    amount / max(s["mean"], 1.), (amount-s["mean"]) / max(std, 1.),
                    len(s["recent"]), len(s["peers"]), pair,
                    pair / max(s["n"], 1), 1 - s["hours"][timestamp.hour] / max(s["n"], 1),
                    receivers.get((r["sender_id"], r["payment_currency"]), {}).get("n", 0),
                ])))
                records.append(r)
            # Equal-timestamp transactions cannot observe each other.
            for r in batch.to_dict("records"):
                for account, amount in [
                    (senders[(r["sender_id"], r["payment_currency"])], r["amount_paid"]),
                    (receivers[(r["receiver_id"], r["receiving_currency"])], r["amount_received"]),
                ]:
                    account["n"] += 1
                    delta = amount - account["mean"]
                    account["mean"] += delta / account["n"]
                    account["m2"] = max(0., account["m2"] + delta * (amount-account["mean"]))
                    account["max"] = max(account["max"], amount)
                s = senders[(r["sender_id"], r["payment_currency"])]
                s["recent"].append(timestamp)
                s["hours"][timestamp.hour] += 1
                s["peers"][r["receiver_id"]] = s["peers"].get(r["receiver_id"], 0) + 1
        return pd.DataFrame(records)
