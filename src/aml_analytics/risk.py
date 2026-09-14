"""Transparent account prioritization, independent of laundering labels."""
import numpy as np


def build_customer_scores(predictions, cutoffs=None):
    scores = predictions.groupby("sender_id").agg(
        transaction_count=("txn_id", "count"),
        total_amount=("amount_paid", "sum"),
        currency_count=("payment_currency", "nunique"),
        max_risk_score=("risk_score", "max"),
        mean_risk_score=("risk_score", "mean"),
        flagged_transactions=("model_flag", "sum"),
        number_of_counterparties=("receiver_id", "nunique"),
        labelled_laundering=("is_laundering", "sum"),
    ).reset_index()
    scores["flagged_proportion"] = scores.flagged_transactions / scores.transaction_count
    # A combined value across currencies has no valid monetary interpretation.
    scores.loc[scores.currency_count > 1, "total_amount"] = np.nan
    if cutoffs is None:
        cutoffs = scores.max_risk_score.quantile([.5, .9]).tolist()
    low, high = cutoffs
    scores["risk_tier"] = np.select(
        [scores.max_risk_score > high, scores.max_risk_score > low],
        ["High", "Medium"], default="Low")
    return scores.sort_values("max_risk_score", ascending=False), [float(low), float(high)]
