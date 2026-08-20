import numpy as np
import pandas as pd

FEATURE_COLS = [
    "amount_log",
    "month",
    "day_of_week",
    "is_month_end",
    "is_year_end",
    "payment_terms_encoded",
    "business_code_encoded",
    "cust_avg_days_late",
    "cust_std_days_late",
    "cust_invoice_count",
    "cust_min_late",
    "cust_max_late",
    "cust_avg_amount",
]


def add_features(df, label_encoders, customer_history, global_defaults):
    """Enrich invoice DataFrame with prediction features.

    Returns the original df with all feature columns added.
    Customer history columns (e.g. cust_std_days_late) are included
    so the decision engine can use them downstream.
    """
    out = df.copy()

    # --- Amount ---
    out["amount_log"] = np.log1p(out["total_open_amount"].abs())

    # --- Temporal (from posting_date) ---
    dates = pd.to_datetime(out["posting_date"])
    out["month"]       = dates.dt.month
    out["day_of_week"] = dates.dt.dayofweek
    out["is_month_end"] = (dates.dt.day > 25).astype(int)
    out["is_year_end"]  = (dates.dt.month == 12).astype(int)

    # --- Categorical encoding ---
    out["payment_terms_encoded"] = (
        out["cust_payment_terms"]
        .map(label_encoders["payment_terms"])
        .fillna(-1)
        .astype(int)
    )
    out["business_code_encoded"] = (
        out["business_code"]
        .map(label_encoders["business_code"])
        .fillna(-1)
        .astype(int)
    )

    # --- Customer history (left join, fall back to global defaults) ---
    hist_cols = [
        "cust_avg_days_late", "cust_std_days_late", "cust_invoice_count",
        "cust_min_late", "cust_max_late", "cust_avg_amount",
    ]
    out = out.merge(customer_history[["cust_number"] + hist_cols],
                    on="cust_number", how="left")

    for col in hist_cols:
        out[col] = out[col].fillna(global_defaults[col])

    return out
