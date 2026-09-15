from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from aml_analytics.modeling import summarize_threshold, evaluate_probability_model
from aml_analytics.risk import build_customer_scores

st.set_page_config(page_title="AML Transaction Monitoring", layout="wide")
st.title("AML Transaction Monitoring & Client Risk Scoring")
st.info("This portfolio project uses synthetic data and is not a production AML decisioning system. Model alerts indicate transactions for further review, not proof of money laundering.")
local_reports = Path(__file__).resolve().parents[1] / "reports"
snapshot = Path(__file__).resolve().parent / "dashboard_data"
reports = local_reports if (local_reports / "predictions.csv").exists() else snapshot

def report_path(name):
    plain = reports / name
    return plain if plain.exists() else reports / (name + ".gz")

if not all(report_path(f).exists() for f in ["metrics.json", "predictions.csv", "customer_risk_scores.csv"]):
    st.warning("Run python scripts/train_model.py --data data/raw/demo_ibm_aml.csv")
    st.stop()
metrics = json.loads((reports / "metrics.json").read_text())
st.caption("Data source: " + metrics.get("dataset", "Unverified legacy reports: retrain before use."))
if reports == snapshot:
    st.caption("Published snapshot of verified results. This app does not train models or process live transactions.")
st.caption(f"Selected model: {metrics['selected_model']} · Operating threshold: {metrics['threshold']:.6f}. Model and threshold selected on validation; KPIs below use filtered test data.")
pred = pd.read_csv(report_path("predictions.csv"), parse_dates=["timestamp"], float_precision="round_trip")
cust = pd.read_csv(report_path("customer_risk_scores.csv"), float_precision="round_trip")
formats = st.sidebar.multiselect("Payment format", sorted(pred.payment_format.unique()), default=sorted(pred.payment_format.unique()))
tiers = st.sidebar.multiselect("Account risk tier", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
dates = st.sidebar.date_input("Date range", value=(pred.timestamp.min().date(), pred.timestamp.max().date()))
pred = pred[pred.payment_format.isin(formats) & pred.sender_id.isin(cust[cust.risk_tier.isin(tiers)].sender_id)]
if len(dates) == 2:
    pred = pred[pred.timestamp.dt.date.between(*dates)]
if pred.empty:
    st.warning("No transactions match these filters.")
    st.stop()
m = summarize_threshold(pred.is_laundering, pred.risk_score.to_numpy(), metrics["threshold"])
ap, _ = evaluate_probability_model(pred.is_laundering, pred.risk_score.to_numpy())
labels = ["Transactions evaluated", "Alerts generated", "Alert rate", "Recall", "Precision", "False-positive rate", "PR-AUC (average precision)"]
values = [f"{len(pred):,}", f"{m['number_of_alerts']:,}", f"{m['alert_rate']:.1%}",
          f"{m['recall']:.1%}" if pred.is_laundering.sum() else "N/A",
          f"{m['precision']:.1%}" if m["number_of_alerts"] else "N/A",
          f"{m['false_positive_rate']:.1%}" if (pred.is_laundering == 0).any() else "N/A",
          f"{ap:.4f}" if ap is not None else "N/A"]
for col, label, value in zip(st.columns(7), labels, values):
    col.metric(label, value)
st.subheader("Risk Score Distribution")
st.plotly_chart(px.histogram(pred, x="risk_score", nbins=50, labels={"risk_score": "Model risk score"}), width="stretch")
st.subheader("Alert Trend")
daily = pred.assign(Date=pred.timestamp.dt.date).groupby("Date", as_index=False).agg(Alerts=("model_flag", "sum"))
st.plotly_chart(px.line(daily, x="Date", y="Alerts"), width="stretch")
st.subheader("Alerts by Payment Format")
fmt = pred.groupby("payment_format", as_index=False).model_flag.sum()
st.plotly_chart(px.bar(fmt, x="payment_format", y="model_flag", labels={"payment_format": "Payment format", "model_flag": "Alerts"}), width="stretch")
def display(frame):
    st.dataframe(frame.rename(columns=lambda c: c.replace("_", " ").title()), width="stretch")
st.subheader("Highest-Risk Accounts")
filtered, _ = build_customer_scores(pred, metrics.get("risk_tier_cutoffs"))
display(filtered.head(30))
st.caption("Activity summaries reflect filters; tiers use fixed validation cutoffs. Total amount is blank for accounts using multiple payment currencies.")
st.subheader("Recent Model Alerts")
display(pred[pred.model_flag == 1].sort_values("timestamp", ascending=False)[
    ["timestamp", "sender_id", "receiver_id", "amount_paid", "payment_currency", "payment_format", "risk_score"]].head(100))
st.subheader("Threshold / False-Positive Trade-off")
st.caption("Full validation period, independent of sidebar filters. The 80% validation recall target does not guarantee future recall.")
if report_path("threshold_curve.csv").exists():
    curve = pd.read_csv(report_path("threshold_curve.csv"), float_precision="round_trip")
    st.plotly_chart(px.line(curve, x="threshold", y=["precision", "recall", "false_positive_rate", "alert_rate"]), width="stretch")
    display(curve)
st.subheader("Model Performance")
st.caption("Model comparison uses validation data. The selected model's test metrics appear above.")
if report_path("model_comparison.csv").exists():
    display(pd.read_csv(report_path("model_comparison.csv")))
display(pd.DataFrame([{k: m[k] for k in ["true_positives", "false_positives", "false_negatives", "true_negatives"]}]))
st.subheader("Feature Importance / Key Risk Drivers")
st.caption("Positive logistic coefficients increase the model score; negative coefficients decrease it. Amount deviation, velocity and repeated counterparties may influence prioritization. Importance describes model associations, not causes or proof of laundering. Scores are not calibrated probabilities.")
if report_path("feature_importance.csv").exists():
    importance = pd.read_csv(report_path("feature_importance.csv"))
    chosen = st.selectbox("Explain model", importance.model.unique())
    subset = importance[importance.model == chosen]
    display(subset.loc[subset.importance.abs().sort_values(ascending=False).index].head(25))
    st.caption("Random forest impurity importance can favor continuous features. HistGradientBoosting has no native importance export; no importance is invented.")
