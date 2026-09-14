from pathlib import Path
import importlib.util
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import duckdb
from aml_analytics.data import load_ibm_transactions, temporal_split
from aml_analytics.features import AMLFeatureEngineer
from aml_analytics.modeling import choose_threshold, summarize_threshold
from aml_analytics.risk import build_customer_scores


def sample():
    return pd.DataFrame({
        "txn_id": ["a", "b", "c"], "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-02"]),
        "sender_id": ["s"]*3, "receiver_id": ["r"]*3,
        "from_bank": ["1"]*3, "to_bank": ["2"]*3,
        "amount_paid": [10., 20., 9999.], "amount_received": [10., 20., 9999.],
        "payment_currency": ["USD"]*3, "receiving_currency": ["USD"]*3,
        "payment_format": ["Wire"]*3, "is_laundering": [0, 1, 0],
    })


def test_history_excludes_current_future_and_ties():
    df = sample()
    fe = AMLFeatureEngineer()
    result = fe.fit_transform(df)
    assert result.sender_txn_count_hist.tolist() == [0, 1, 1]
    assert result.sender_amount_mean_hist.tolist() == [0, 10, 10]
    changed = df.copy()
    changed.loc[2, "amount_paid"] = 1e8
    pd.testing.assert_frame_equal(result.iloc[:2], AMLFeatureEngineer().fit_transform(changed).iloc[:2])
    fe.fit(df.iloc[:1])
    first = fe.transform(df.iloc[1:])
    pd.testing.assert_frame_equal(first, fe.transform(df.iloc[1:]))
    assert first.sender_txn_count_hist.tolist() == [1, 1]


def test_currency_history_and_temporal_boundary():
    df = sample()
    df.loc[1, "payment_currency"] = "CAD"
    assert AMLFeatureEngineer().fit_transform(df).sender_txn_count_hist.tolist() == [0, 0, 1]
    train, test = temporal_split(df)
    assert train.timestamp.max() < test.timestamp.min()


def test_exact_threshold_including_tied_and_low_scores():
    y, prob = pd.Series([1, 0, 1, 0]), np.array([.001, .001, .0001, 0])
    threshold, curve = choose_threshold(y, prob, 1)
    assert threshold == .0001
    assert summarize_threshold(y, prob, threshold)["number_of_alerts"] == 3
    for row in curve.to_dict("records"):
        expected = summarize_threshold(y, prob, row["threshold"])
        for key in expected:
            assert row[key] == pytest.approx(expected[key])
    with pytest.raises(ValueError):
        choose_threshold(pd.Series([0, 0]), np.array([.1, .2]))


def test_risk_labels_do_not_affect_tiers():
    df = sample().assign(risk_score=[.1, .4, .8], model_flag=[0, 0, 1])
    scores, cutoffs = build_customer_scores(df)
    other, _ = build_customer_scores(df.assign(is_laundering=1-df.is_laundering), cutoffs)
    assert scores.risk_tier.tolist() == other.risk_tier.tolist()
    assert scores.flagged_proportion.iloc[0] == pytest.approx(1/3)
    df.loc[1, "payment_currency"] = "CAD"
    assert build_customer_scores(df)[0].total_amount.isna().all()


def test_loader_preserves_ids_and_rejects_bad_labels(tmp_path):
    df = pd.DataFrame([["2025/01/01 00:00", "001", "0002", "003", "0004",
                        10, "USD", 10, "USD", "Wire", 0]],
                      columns=["Timestamp", "From Bank", "Account", "To Bank", "Account.1",
                               "Amount Received", "Receiving Currency", "Amount Paid",
                               "Payment Currency", "Payment Format", "Is Laundering"])
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    assert load_ibm_transactions(path).sender_id.iloc[0] == "001::0002"
    df["Is Laundering"] = 2
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_ibm_transactions(path)


def test_sql_executes():
    transactions = sample()
    with duckdb.connect() as con:
        con.register("transactions", transactions)
        con.execute(Path("sql/01_monitoring_queries.sql").read_text())


def test_download_single_file(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("download", "scripts/download_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cached = tmp_path / "cache" / "HI-Small_Trans.csv"
    cached.parent.mkdir()
    cached.write_text("test fixture")
    monkeypatch.setattr(module, "TARGET_DIR", tmp_path / "raw")
    with patch.object(module.kagglehub, "dataset_download", return_value=str(cached)) as download:
        module.main()
        assert (module.TARGET_DIR / cached.name).read_text() == "test fixture"
        assert download.call_args.kwargs["path"] == "HI-Small_Trans.csv"
