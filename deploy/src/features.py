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

HIST_COLS = [
    "cust_avg_days_late", "cust_std_days_late", "cust_invoice_count",
    "cust_min_late", "cust_max_late", "cust_avg_amount",
]


def build_customer_timeline(closed):
    """Running per-customer aggregates of days_late, stamped with clear_date.

    Each row is what was knowable about a customer the moment their nth invoice
    cleared. Joined as-of posting_date (see add_features), so a prediction can
    only ever see outcomes that had already happened when the invoice was raised.

    Replaces the previous static per-customer table, which aggregated all 40k
    closed invoices at once and therefore leaked each row's own target — and
    every future invoice's — into its features.
    """
    d = closed.sort_values("clear_date").reset_index(drop=True)
    late = d.groupby("cust_number")["days_late"]
    amount = d.groupby("cust_number")["total_open_amount"]

    def expand(grouped, how):
        return getattr(grouped.expanding(), how)().reset_index(level=0, drop=True)

    return pd.DataFrame({
        "cust_number":        d["cust_number"],
        "clear_date":         d["clear_date"],
        "cust_avg_days_late": expand(late, "mean"),
        "cust_std_days_late": expand(late, "std"),
        "cust_invoice_count": expand(late, "count"),
        "cust_min_late":      expand(late, "min"),
        "cust_max_late":      expand(late, "max"),
        "cust_avg_amount":    expand(amount, "mean"),
    }).sort_values("clear_date").reset_index(drop=True)


def latest_customer_state(timeline):
    """Final row per customer — the descriptive snapshot used by the segmentation chart.

    Not used for model features: it is the full-history view by construction.
    """
    return (timeline.sort_values("clear_date")
                    .groupby("cust_number", as_index=False)
                    .last()
                    .drop(columns=["clear_date"]))


def add_features(df, label_encoders, customer_timeline, global_defaults):
    """Enrich invoice DataFrame with prediction features.

    Customer history is joined as-of the invoice's posting_date, so invoices
    from a customer with no cleared history yet fall back to global defaults.
    """
    out = df.copy()

    # --- Amount ---
    out["amount_log"] = np.log1p(out["total_open_amount"].abs())

    # --- Temporal (from posting_date) ---
    dates = pd.to_datetime(out["posting_date"])
    out["posting_date"] = dates
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

    # --- Customer history as of posting_date (strictly before, never same-day) ---
    # Renamed so the join key cannot collide with a clear_date already on df.
    timeline = customer_timeline.rename(columns={"clear_date": "_hist_asof"})
    out["_order"] = np.arange(len(out))
    merged = pd.merge_asof(
        out.sort_values("posting_date"),
        timeline,
        left_on="posting_date",
        right_on="_hist_asof",
        by="cust_number",
        direction="backward",
        allow_exact_matches=False,
    )
    out = merged.sort_values("_order").drop(columns=["_order", "_hist_asof"]).reset_index(drop=True)

    for col in HIST_COLS:
        out[col] = out[col].fillna(global_defaults[col])

    return out
