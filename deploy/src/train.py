"""Train the payment-lateness models and write every artifact the app loads.

Replaces archive/scripts/train_model.py, which imported the deleted app.* package
and could no longer run. Two things changed versus that script:

  1. Customer history is point-in-time (see features.build_customer_timeline)
     rather than aggregated over all closed invoices at once.
  2. The train/test split is chronological, matching how the model is actually
     used: fit on the past, predict invoices raised later.

Run: python -m src.train      (from the deploy/ directory)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.features import (
    FEATURE_COLS, HIST_COLS, add_features,
    build_customer_timeline, latest_customer_state,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
TEST_FRACTION = 0.2
GBM = dict(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)


def load_closed():
    df = pd.read_csv(ROOT / "dataset.csv")
    df["posting_date"] = pd.to_datetime(df["posting_date"], format="mixed")
    df["due_in_date"] = pd.to_datetime(df["due_in_date"].astype(str), format="%Y%m%d")
    df["clear_date"] = pd.to_datetime(df["clear_date"], format="mixed", errors="coerce")
    closed = df[df["isOpen"] == 0].copy()
    closed["days_late"] = (closed["clear_date"] - closed["due_in_date"]).dt.days
    return closed.dropna(subset=["days_late", "clear_date"]).reset_index(drop=True)


def chronological_split(closed):
    """Split on posting_date so the test fold is strictly later than the train fold."""
    ordered = closed.sort_values("posting_date").reset_index(drop=True)
    cut = int(len(ordered) * (1 - TEST_FRACTION))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def build_encoders(train):
    return {
        "payment_terms": {v: i for i, v in enumerate(sorted(train["cust_payment_terms"].dropna().unique()))},
        "business_code": {v: i for i, v in enumerate(sorted(train["business_code"].dropna().unique()))},
    }


def build_defaults(train):
    """Fallbacks for customers with no cleared invoice yet. Train fold only."""
    return {
        "cust_avg_days_late": float(train["days_late"].mean()),
        "cust_std_days_late": float(train["days_late"].std()),
        "cust_invoice_count": 0.0,
        "cust_min_late": float(train["days_late"].min()),
        "cust_max_late": float(train["days_late"].max()),
        "cust_avg_amount": float(train["total_open_amount"].mean()),
    }


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    closed = load_closed()
    train, test = chronological_split(closed)
    print(f"Closed invoices: {len(closed)}")
    print(f"  train: {len(train)}  ({train.posting_date.min().date()} .. {train.posting_date.max().date()})")
    print(f"  test:  {len(test)}  ({test.posting_date.min().date()} .. {test.posting_date.max().date()})")

    encoders = build_encoders(train)
    defaults = build_defaults(train)

    # The timeline is built from the train fold only, so no test outcome can reach
    # a training feature. At serving time it is rebuilt over all closed invoices —
    # by then those outcomes are genuinely known.
    train_timeline = build_customer_timeline(train)

    ftr = add_features(train, encoders, train_timeline, defaults)
    fte = add_features(test, encoders, train_timeline, defaults)
    X_train, y_train = ftr[FEATURE_COLS].values, ftr["days_late"].values
    X_test, y_test = fte[FEATURE_COLS].values, fte["days_late"].values

    print("Training mean / lower(P10) / upper(P90) models...")
    model_mean = GradientBoostingRegressor(**GBM).fit(X_train, y_train)
    model_lower = GradientBoostingRegressor(loss="quantile", alpha=0.1, **GBM).fit(X_train, y_train)
    model_upper = GradientBoostingRegressor(loss="quantile", alpha=0.9, **GBM).fit(X_train, y_train)

    y_pred = model_mean.predict(X_test)
    y_lower = model_lower.predict(X_test)
    y_upper = model_upper.predict(X_test)

    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    within_3 = float(np.mean(np.abs(y_test - y_pred) <= 3))
    coverage = float(np.mean((y_test >= y_lower) & (y_test <= y_upper)))

    print("\n=== Held-out performance (chronological test fold) ===")
    print(f"MAE:             {mae:.2f} days")
    print(f"RMSE:            {rmse:.2f} days")
    print(f"Within +/-3 days: {within_3 * 100:.1f}%")
    print(f"80% PI coverage: {coverage * 100:.1f}%")

    hist_share = sum(model_mean.feature_importances_[FEATURE_COLS.index(c)] for c in HIST_COLS)
    print(f"Importance carried by customer-history features: {hist_share * 100:.1f}%")

    # Serving artifacts: timeline over ALL closed invoices, encoders from the full set
    # so unseen payment terms in the open book still map to a real code.
    serve_timeline = build_customer_timeline(closed)
    joblib.dump(model_mean, ARTIFACTS / "model_mean.joblib")
    joblib.dump(model_lower, ARTIFACTS / "model_lower.joblib")
    joblib.dump(model_upper, ARTIFACTS / "model_upper.joblib")
    joblib.dump(build_encoders(closed), ARTIFACTS / "label_encoders.joblib")
    joblib.dump(serve_timeline, ARTIFACTS / "customer_timeline.joblib")
    joblib.dump(latest_customer_state(serve_timeline), ARTIFACTS / "customer_history.joblib")
    joblib.dump(defaults, ARTIFACTS / "global_defaults.joblib")

    write_test_results(fte, y_pred, y_lower, y_upper, y_test,
                       dict(mae=mae, rmse=rmse, within_3=within_3, coverage=coverage))
    print(f"\nArtifacts written to {ARTIFACTS}")


def write_test_results(fte, y_pred, y_lower, y_upper, y_test, m):
    """Dashboard payload. Only the metrics and invoices keys are read by analytics.py."""
    invoices = [{
        "invoice_id": str(r.invoice_id),
        "cust_number": str(r.cust_number),
        "name_customer": str(r.name_customer),
        "invoice_amount": round(float(r.total_open_amount), 2),
        "predicted_days_late": round(float(p), 1),
        "prediction_lower": round(float(lo), 1),
        "prediction_upper": round(float(hi), 1),
        "actual_days_late": round(float(a), 1),
        "cust_std_days_late": round(float(r.cust_std_days_late), 2),
    } for r, p, lo, hi, a in zip(fte.itertuples(), y_pred, y_lower, y_upper, y_test)]

    payload = {
        "metrics": {
            "rmse": round(m["rmse"], 2),
            "mae": round(m["mae"], 2),
            "within_3_days": round(m["within_3"] * 100, 1),
            "pi_coverage": round(m["coverage"] * 100, 1),
        },
        "invoices": invoices,
        "split": "chronological",
    }
    with open(ARTIFACTS / "test_results.json", "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
