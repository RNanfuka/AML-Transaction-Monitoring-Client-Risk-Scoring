import pandas as pd

from aml_analytics.features import AMLFeatureEngineer


def test_feature_engineer_outputs_expected_columns():
    df = pd.DataFrame(
        {
            "txn_id": ["T1", "T2", "T3"],
            "timestamp": pd.to_datetime(
                ["2025-01-01 10:00", "2025-01-02 11:00", "2025-01-03 12:00"]
            ),
            "from_bank": ["1", "1", "2"],
            "sender_account": ["A", "A", "B"],
            "to_bank": ["2", "3", "1"],
            "receiver_account": ["C", "D", "A"],
            "amount_received": [100.0, 120.0, 5000.0],
            "receiving_currency": ["USD", "USD", "CAD"],
            "amount_paid": [100.0, 120.0, 5000.0],
            "payment_currency": ["USD", "USD", "USD"],
            "payment_format": ["ACH", "Wire", "Wire"],
            "is_laundering": [0, 0, 1],
            "sender_id": ["1::A", "1::A", "2::B"],
            "receiver_id": ["2::C", "3::D", "1::A"],
        }
    )

    fe = AMLFeatureEngineer()
    out = fe.fit_transform(df)

    for col in AMLFeatureEngineer.numeric_features:
        assert col in out.columns
    for col in AMLFeatureEngineer.categorical_features:
        assert col in out.columns

    assert out["currency_mismatch"].sum() == 1
    assert out["cross_bank"].sum() == 3
