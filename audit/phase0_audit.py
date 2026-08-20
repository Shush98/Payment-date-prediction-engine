"""Phase 0 audit: reproduce the baseline, then measure how much of it is leakage.

The original training script (archive/scripts/train_model.py) imports app.config and
app.models.* which no longer exist, so the pipeline that produced deploy/artifacts/
cannot be re-run. This reconstructs it from the surviving artifact schemas, then
re-trains under progressively stricter conditions to isolate the leakage.

Run: python audit/phase0_audit.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
HIST_COLS = ["cust_avg_days_late", "cust_std_days_late", "cust_invoice_count",
             "cust_min_late", "cust_max_late", "cust_avg_amount"]
FEATURE_COLS = ["amount_log", "month", "day_of_week", "is_month_end", "is_year_end",
                "payment_terms_encoded", "business_code_encoded"] + HIST_COLS
GBM = dict(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)


def load_closed():
    df = pd.read_csv(ROOT / "deploy" / "dataset.csv")
    df["posting_date"] = pd.to_datetime(df["posting_date"], format="mixed")
    df["due_in_date"] = pd.to_datetime(df["due_in_date"].astype(str), format="%Y%m%d")
    df["clear_date"] = pd.to_datetime(df["clear_date"], format="mixed", errors="coerce")
    closed = df[df["isOpen"] == 0].copy()
    closed["days_late"] = (closed["clear_date"] - closed["due_in_date"]).dt.days
    return closed.dropna(subset=["days_late"]).reset_index(drop=True)


def build_history(src):
    """Per-customer aggregates of the target. This is the leakage surface."""
    g = src.groupby("cust_number")
    h = pd.DataFrame({
        "cust_avg_days_late": g["days_late"].mean(),
        "cust_std_days_late": g["days_late"].std(),
        "cust_invoice_count": g["days_late"].count(),
        "cust_min_late": g["days_late"].min(),
        "cust_max_late": g["days_late"].max(),
        "cust_avg_amount": g["total_open_amount"].mean(),
    }).reset_index()
    return h.fillna({"cust_std_days_late": 0.0})


def loo_history(src):
    """Leave-one-out: each row's customer aggregates exclude that row's own target."""
    g = src.groupby("cust_number")["days_late"]
    n = g.transform("count")
    s = g.transform("sum")
    ss = g.transform(lambda x: (x ** 2).sum())
    n_o = (n - 1).clip(lower=1)
    mean_o = (s - src["days_late"]) / n_o
    var_o = ((ss - src["days_late"] ** 2) / n_o - mean_o ** 2).clip(lower=0)
    out = pd.DataFrame(index=src.index)
    out["cust_avg_days_late"] = np.where(n > 1, mean_o, 0.0)
    out["cust_std_days_late"] = np.where(n > 1, np.sqrt(var_o), 0.0)
    out["cust_invoice_count"] = n - 1
    # min/max excluding self are expensive; rank-based approximation is not worth it here.
    # ponytail: reuse full-sample min/max, so this UNDERSTATES the leakage removal.
    gg = src.groupby("cust_number")["days_late"]
    out["cust_min_late"] = gg.transform("min")
    out["cust_max_late"] = gg.transform("max")
    out["cust_avg_amount"] = src.groupby("cust_number")["total_open_amount"].transform("mean")
    return out


def featurize(df, encoders, history, defaults):
    out = df.copy()
    out["amount_log"] = np.log1p(out["total_open_amount"].abs())
    d = out["posting_date"]
    out["month"] = d.dt.month
    out["day_of_week"] = d.dt.dayofweek
    out["is_month_end"] = (d.dt.day > 25).astype(int)
    out["is_year_end"] = (d.dt.month == 12).astype(int)
    out["payment_terms_encoded"] = out["cust_payment_terms"].map(encoders["pt"]).fillna(-1).astype(int)
    out["business_code_encoded"] = out["business_code"].map(encoders["bc"]).fillna(-1).astype(int)
    if history is not None:
        out = out.merge(history[["cust_number"] + HIST_COLS], on="cust_number", how="left")
        for c in HIST_COLS:
            out[c] = out[c].fillna(defaults[c])
    return out


def make_encoders(src):
    return {"pt": {v: i for i, v in enumerate(sorted(src["cust_payment_terms"].dropna().unique()))},
            "bc": {v: i for i, v in enumerate(sorted(src["business_code"].dropna().unique()))}}


def evaluate(Xtr, ytr, Xte, yte, quantiles=False):
    m = GradientBoostingRegressor(**GBM).fit(Xtr, ytr)
    p = m.predict(Xte)
    r = {"MAE": np.mean(np.abs(yte - p)),
         "RMSE": np.sqrt(np.mean((yte - p) ** 2)),
         "within_3d": np.mean(np.abs(yte - p) <= 3) * 100}
    if quantiles:
        lo = GradientBoostingRegressor(loss="quantile", alpha=0.1, **GBM).fit(Xtr, ytr).predict(Xte)
        hi = GradientBoostingRegressor(loss="quantile", alpha=0.9, **GBM).fit(Xtr, ytr).predict(Xte)
        r["PI_cov"] = np.mean((yte >= lo) & (yte <= hi)) * 100
    return r


def main():
    closed = load_closed()
    enc = make_encoders(closed)
    defaults = {c: (closed["days_late"].mean() if c == "cust_avg_days_late"
                    else closed["days_late"].std() if c == "cust_std_days_late"
                    else closed["total_open_amount"].mean() if c == "cust_avg_amount"
                    else 0) for c in HIST_COLS}
    rows = {}

    # A — as shipped: history over ALL closed rows, random split.
    f = featurize(closed, enc, build_history(closed), defaults)
    Xtr, Xte, ytr, yte = train_test_split(f[FEATURE_COLS].values, f["days_late"].values,
                                          test_size=0.2, random_state=42)
    rows["A. shipped (full history + random split)"] = evaluate(Xtr, ytr, Xte, yte, quantiles=True)

    # B — history built from TRAIN rows only; still random split.
    tr_idx, te_idx = train_test_split(np.arange(len(closed)), test_size=0.2, random_state=42)
    f = featurize(closed, enc, build_history(closed.iloc[tr_idx]), defaults)
    rows["B. history from train only (random split)"] = evaluate(
        f[FEATURE_COLS].values[tr_idx], closed["days_late"].values[tr_idx],
        f[FEATURE_COLS].values[te_idx], closed["days_late"].values[te_idx])

    # C — leave-one-out history: removes a row's own target from its own features.
    f = featurize(closed, enc, None, defaults)
    for c in HIST_COLS:
        f[c] = loo_history(closed)[c].values
    rows["C. leave-one-out history (random split)"] = evaluate(
        f[FEATURE_COLS].values[tr_idx], closed["days_late"].values[tr_idx],
        f[FEATURE_COLS].values[te_idx], closed["days_late"].values[te_idx])

    # D — deployment reality: train on the past, test on the future.
    order = closed.sort_values("posting_date").index.to_numpy()
    cut = int(len(order) * 0.8)
    tr_i, te_i = order[:cut], order[cut:]
    f = featurize(closed, enc, build_history(closed.loc[tr_i]), defaults)
    rows["D. time split + history from past only"] = evaluate(
        f[FEATURE_COLS].values[tr_i], closed["days_late"].values[tr_i],
        f[FEATURE_COLS].values[te_i], closed["days_late"].values[te_i], quantiles=True)

    print(f"\n{'variant':<44}{'MAE':>8}{'RMSE':>8}{'±3d %':>9}{'PI cov %':>10}")
    print("-" * 79)
    for k, v in rows.items():
        print(f"{k:<44}{v['MAE']:>8.2f}{v['RMSE']:>8.2f}{v['within_3d']:>9.1f}"
              f"{v.get('PI_cov', float('nan')):>10.1f}")
    base, real = rows["A. shipped (full history + random split)"], rows["D. time split + history from past only"]
    print(f"\nMAE inflation from A to D: {real['MAE'] / base['MAE']:.2f}x "
          f"({base['MAE']:.2f} -> {real['MAE']:.2f} days)")


if __name__ == "__main__":
    main()
