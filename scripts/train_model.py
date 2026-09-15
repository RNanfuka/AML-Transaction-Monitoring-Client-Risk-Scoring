from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from aml_analytics.data import load_ibm_transactions, temporal_split
from aml_analytics.features import AMLFeatureEngineer
from aml_analytics.modeling import (
    build_models, choose_threshold, evaluate_probability_model, save_metrics,
    save_model_bundle, summarize_threshold, feature_importance,
)
from aml_analytics.risk import build_customer_scores


def cap(df, limit, seed):
    """Choose output rows uniformly, independently of labels; replay all history."""
    return df.sample(n=min(len(df), limit), random_state=seed).sort_values("timestamp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/HI-Small_Trans.csv")
    parser.add_argument("--max-train-rows", type=int, default=150000)
    parser.add_argument("--max-test-rows", type=int, default=60000)
    parser.add_argument("--target-recall", type=float, default=.80)
    args = parser.parse_args()
    if min(args.max_train_rows, args.max_test_rows) < 1 or not 0 < args.target_recall <= 1:
        parser.error("Row limits must be positive and target recall must be in (0, 1].")
    if not Path(args.data).exists():
        parser.error("Download HI-Small_Trans.csv into data/raw/ or use --data data/raw/demo_ibm_aml.csv.")
    print("Loading and validating transactions...", flush=True)
    raw = load_ibm_transactions(args.data)
    print(f"Validated {len(raw):,} rows.", flush=True)
    development, test_raw = temporal_split(raw, .8)
    train_raw, validation_raw = temporal_split(development, .75)
    train_ids = set(cap(train_raw, args.max_train_rows, 42).txn_id)
    validation_ids = set(cap(validation_raw, args.max_test_rows, 43).txn_id)
    test_ids = set(cap(test_raw, args.max_test_rows, 44).txn_id)
    engineer = AMLFeatureEngineer()
    print("Building training history; retaining sampled features...", flush=True)
    train = engineer.fit_transform(train_raw, selected_ids=train_ids)
    # Replay every row, but retain only the label-independent evaluation sample.
    print("Replaying validation/test history...", flush=True)
    future = engineer.transform(pd.concat([validation_raw, test_raw]),
                                selected_ids=validation_ids | test_ids)
    validation = future[future.txn_id.isin(validation_ids)]
    test = future[future.txn_id.isin(test_ids)]
    for label, split in [("training", train), ("validation", validation), ("test", test)]:
        if split.is_laundering.nunique() != 2:
            raise ValueError(f"{label} requires both classes; increase row limits or supply more data.")
    cols = engineer.numeric_features + engineer.categorical_features
    models = build_models()
    comparison = []
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        model.fit(train[cols], train.is_laundering)
        prob = model.predict_proba(validation[cols])[:, 1]
        ap, auc = evaluate_probability_model(validation.is_laundering, prob)
        threshold, _ = choose_threshold(validation.is_laundering, prob, args.target_recall)
        comparison.append(dict(model=name, pr_auc=ap, roc_auc=auc,
                               **summarize_threshold(validation.is_laundering, prob, threshold)))
    comparison = pd.DataFrame(comparison).sort_values(
        ["pr_auc", "number_of_alerts"], ascending=[False, True])
    best = comparison.iloc[0]
    name, threshold = best.model, float(best.threshold)
    model = models[name]
    vp = model.predict_proba(validation[cols])[:, 1]
    _, curve = choose_threshold(validation.is_laundering, vp, args.target_recall)
    prob = model.predict_proba(test[cols])[:, 1]
    ap, auc = evaluate_probability_model(test.is_laundering, prob)
    val_out = validation.assign(risk_score=vp, model_flag=(vp >= threshold).astype(int))
    _, cutoffs = build_customer_scores(val_out)
    pred = test.assign(risk_score=prob, model_flag=(prob >= threshold).astype(int))
    customer_scores, _ = build_customer_scores(pred, cutoffs)
    source = "IBM HI-Small (synthetic)" if Path(args.data).name == "HI-Small_Trans.csv" else "DEMO / unverified input — not portfolio metrics"
    summary = dict(rows=len(raw), laundering_rows=int(raw.is_laundering.sum()),
                   laundering_rate=float(raw.is_laundering.mean()),
                   date_start=str(raw.timestamp.min()), date_end=str(raw.timestamp.max()),
                   unique_senders=int(raw.sender_id.nunique()), unique_receivers=int(raw.receiver_id.nunique()))
    metrics = dict(dataset=source, data_path=str(Path(args.data)), selected_model=name,
                   pr_auc=ap, roc_auc=auc, target_recall=args.target_recall,
                   **summarize_threshold(test.is_laundering, prob, threshold),
                   train_rows=len(train), validation_rows=len(validation), test_rows=len(test),
                   test_positive_rate=float(test.is_laundering.mean()), risk_tier_cutoffs=cutoffs,
                   selection_period="validation", evaluation_period="held-out test",
                   dataset_summary=summary)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    save_model_bundle("models/aml_model.joblib", model, engineer, threshold, name)
    save_metrics(reports / "metrics.json", metrics)
    save_metrics(reports / "dataset_summary.json", summary)
    curve.assign(period="validation").to_csv(reports / "threshold_curve.csv", index=False)
    comparison.assign(period="validation").to_csv(reports / "model_comparison.csv", index=False)
    feature_importance(models).to_csv(reports / "feature_importance.csv", index=False)
    pred.to_csv(reports / "predictions.csv", index=False)
    customer_scores.to_csv(reports / "customer_risk_scores.csv", index=False)
    print(pd.Series(metrics).to_string())
    print("Reports replaced by this run. The validation recall target is not a test recall guarantee.")


if __name__ == "__main__":
    main()
