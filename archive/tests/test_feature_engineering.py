import numpy as np
import pandas as pd
import pytest
from app.models.feature_engineering import engineer_features, compute_customer_responsiveness


@pytest.fixture
def sample_training_data():
    return pd.DataFrame({
        "total_open_amount": [1000, 2000, 500, 1500],
        "posting_date": pd.to_datetime(["2020-01-15", "2020-06-25", "2020-12-01", "2020-03-10"]),
        "cust_payment_terms": ["NAA8", "NAH4", "NAA8", "NAD1"],
        "business_code": ["U001", "CA02", "U001", "CA02"],
        "cust_number": ["C1", "C2", "C1", "C3"],
        "days_late": [3, -2, 5, 0],
    })


def test_training_mode_returns_four_items(sample_training_data):
    result = engineer_features(sample_training_data)
    assert len(result) == 4
    features, encoders, history, defaults = result
    assert "amount_log" in features.columns
    assert "payment_terms" in encoders
    assert "business_code" in encoders
    assert "cust_avg_days_late" in defaults


def test_feature_columns_present(sample_training_data):
    features, _, _, _ = engineer_features(sample_training_data)
    expected_cols = [
        "amount_log", "month", "day_of_week", "is_month_end", "is_year_end",
        "payment_terms_encoded", "business_code_encoded",
        "cust_avg_days_late", "cust_std_days_late", "cust_invoice_count",
        "cust_min_late", "cust_max_late", "cust_avg_amount",
    ]
    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"


def test_inference_mode(sample_training_data):
    _, encoders, history, defaults = engineer_features(sample_training_data)

    new_data = pd.DataFrame({
        "total_open_amount": [3000],
        "posting_date": pd.to_datetime(["2021-01-20"]),
        "cust_payment_terms": ["NAA8"],
        "business_code": ["U001"],
        "cust_number": ["C1"],
    })

    features = engineer_features(new_data, encoders, history, defaults)
    assert isinstance(features, pd.DataFrame)
    assert len(features) == 1
    assert features["payment_terms_encoded"].iloc[0] >= 0


def test_unknown_customer_uses_defaults(sample_training_data):
    _, encoders, history, defaults = engineer_features(sample_training_data)

    new_data = pd.DataFrame({
        "total_open_amount": [1000],
        "posting_date": pd.to_datetime(["2021-01-20"]),
        "cust_payment_terms": ["NAA8"],
        "business_code": ["U001"],
        "cust_number": ["UNKNOWN"],
    })

    features = engineer_features(new_data, encoders, history, defaults)
    assert not np.isnan(features["cust_avg_days_late"].iloc[0])


def test_is_month_end_flag(sample_training_data):
    features, _, _, _ = engineer_features(sample_training_data)
    # posting_date[1] is June 25 -> day > 25 is False (25 is not > 25)
    # posting_date[0] is Jan 15 -> False
    assert features["is_month_end"].dtype in [int, np.int64, np.int32]


def test_is_year_end_flag(sample_training_data):
    features, _, _, _ = engineer_features(sample_training_data)
    # Row index 2 has month=12 -> is_year_end=1
    dec_rows = features[features["month"] == 12]
    assert all(dec_rows["is_year_end"] == 1)


def test_customer_responsiveness():
    df = pd.DataFrame({"cust_std_days_late": [0, 10, 20, 100]})
    resp = compute_customer_responsiveness(df)
    assert resp.iloc[0] == 0.8  # Low std = high responsiveness
    assert resp.iloc[3] == 0.2  # High std = low responsiveness (clipped)
