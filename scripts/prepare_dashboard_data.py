"""Package verified reports for the public dashboard; never include raw data or models."""
from pathlib import Path
import shutil
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports"
TARGET = ROOT / "app" / "dashboard_data"

def main():
    metrics = json.loads((SOURCE / "metrics.json").read_text())
    if metrics.get("dataset") != "IBM HI-Small (synthetic)":
        raise ValueError("Run and verify the IBM pipeline before preparing public results.")
    TARGET.mkdir(exist_ok=True)
    for name in ["metrics.json", "dataset_summary.json"]:
        shutil.copy2(SOURCE / name, TARGET / name)
    columns = ["txn_id", "timestamp", "sender_id", "receiver_id", "amount_paid",
               "payment_currency", "payment_format", "is_laundering", "risk_score", "model_flag"]
    for name in ["predictions", "customer_risk_scores", "threshold_curve",
                 "model_comparison", "feature_importance"]:
        frame = pd.read_csv(SOURCE / f"{name}.csv", float_precision="round_trip",
                            usecols=columns if name == "predictions" else None)
        frame.to_csv(TARGET / f"{name}.csv.gz", index=False,
                     compression={"method": "gzip", "mtime": 0})
    (TARGET / "README.md").write_text(
        "# Published dashboard snapshot\n\n"
        "Derived results from the verified IBM HI-Small synthetic dataset run.\n"
        "Includes all 60,000 evaluated test rows with only dashboard columns, account\n"
        "summaries, validation threshold results and model comparison/importance.\n"
        "No raw training dataset or model binaries are included.\n"
        "This snapshot is static; the hosted app does not train models.\n"
        "Run scripts/prepare_dashboard_data.py after a newly verified training run.\n")
    print(f"Prepared {TARGET}")
    print(f"Total: {sum(p.stat().st_size for p in TARGET.iterdir()):,} bytes")
    print("No files have been staged, committed, pushed or deployed.")

if __name__ == "__main__":
    main()
