import math
import numpy as np
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, jsonify, request

from src.predict import load_models, predict
from src.model_source import load_predictor_deferred
from src.decision import build_priority_table, DAILY_CAPACITY, _p_late, _p_responds
from src.analytics import compute_analytics, compute_strategy_comparison

ROOT_DIR = Path(__file__).parent

app = Flask(__name__)

@app.template_filter("format_number")
def format_number(value):
    return f"{int(value):,}"

# --- Load everything once at startup ---
models = load_models()

# Returns immediately with the bundled artifacts and upgrades to the Unity Catalog model
# in a background thread. Loading a registered model is a network call; doing it inline
# here blocks gunicorn's worker boot, the port never opens, and the host kills the deploy.
PREDICTOR = load_predictor_deferred(models)
print(f"[model] serving {PREDICTOR.source} ({PREDICTOR.version}) - {PREDICTOR.detail}")

raw = pd.read_csv(ROOT_DIR / "dataset.csv")
raw["posting_date"] = pd.to_datetime(raw["posting_date"], format="mixed")
raw["due_in_date"]  = pd.to_datetime(raw["due_in_date"].astype(str), format="%Y%m%d")

OPEN_INVOICES = raw[raw["isOpen"] == 1].reset_index(drop=True)

_today    = pd.Timestamp.today().normalize()
_days_open = (_today - OPEN_INVOICES["posting_date"]).dt.days.clip(lower=0)

STATIC_KPIS = {
    "total_invoices":    len(OPEN_INVOICES),
    "total_outstanding": round(float(OPEN_INVOICES["total_open_amount"].sum()), 2),
    "oldest_days":       int(_days_open.max()),
}

ANALYTICS = compute_analytics(
    models,
    models["customer_history"],
    ROOT_DIR / "artifacts" / "test_results.json",
)


@app.route("/")
def dashboard():
    return render_template("dashboard.html", kpis=STATIC_KPIS)


@app.route("/health")
def health():
    """Liveness plus model provenance. Makes it visible which model is actually serving
    rather than leaving a silent fallback to look like the real thing."""
    # Kicks off the Unity Catalog load on first call, and revives it if the thread ever
    # disappeared. Idempotent and non-blocking.
    PREDICTOR.ensure_started()
    return jsonify({
        "status": "ok",
        "model": PREDICTOR.as_dict(),
        "open_invoices": STATIC_KPIS["total_invoices"],
    })


@app.route("/api/analytics")
def get_analytics():
    return jsonify(ANALYTICS)


@app.route("/api/invoices")
def get_invoices():
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 10))

    total = len(OPEN_INVOICES)
    pages = math.ceil(total / limit)
    start = (page - 1) * limit
    end   = start + limit

    slice_ = OPEN_INVOICES.iloc[start:end][[
        "invoice_id", "name_customer", "total_open_amount", "due_in_date"
    ]].copy()
    slice_["due_in_date"] = slice_["due_in_date"].dt.strftime("%Y-%m-%d")

    return jsonify({
        "invoices": slice_.to_dict("records"),
        "total":    total,
        "page":     page,
        "pages":    pages,
    })


@app.route("/api/predict", methods=["POST"])
def run_predictions():
    PREDICTOR.ensure_started()
    predictions = predict(OPEN_INVOICES, models, PREDICTOR)
    priority    = build_priority_table(predictions)

    # --- Invoice table payload ---
    inv = predictions[[
        "invoice_id", "name_customer", "total_open_amount",
        "due_in_date", "days_late_pred", "days_late_lower", "days_late_upper",
        "predicted_payment_date",
    ]].copy()
    inv["due_in_date"]            = inv["due_in_date"].dt.strftime("%Y-%m-%d")
    inv["predicted_payment_date"] = inv["predicted_payment_date"].dt.strftime("%Y-%m-%d")
    inv["days_late_pred"]         = np.ceil(inv["days_late_pred"]).astype(int)

    # Confidence range dates: due_date + ceil(lower/upper)
    due_dt = pd.to_datetime(predictions["due_in_date"])
    inv["predicted_payment_lower"] = (
        due_dt + pd.to_timedelta(np.ceil(predictions["days_late_lower"]).astype(int), unit="D")
    ).dt.strftime("%Y-%m-%d")
    inv["predicted_payment_upper"] = (
        due_dt + pd.to_timedelta(np.ceil(predictions["days_late_upper"]).astype(int), unit="D")
    ).dt.strftime("%Y-%m-%d")

    inv["p_late"] = [
        round(_p_late(r["days_late_lower"], r["days_late_upper"]), 2)
        for _, r in predictions.iterrows()
    ]
    inv["days_late_lower"] = np.ceil(inv["days_late_lower"]).astype(int)
    inv["days_late_upper"] = np.ceil(inv["days_late_upper"]).astype(int)

    # --- Priority table payload (top 100) ---
    pri = priority[[
        "rank", "name_customer", "total_open_amount",
        "days_late_pred", "days_late_lower", "days_late_upper",
        "expected_value", "action",
        "cust_std_days_late",
    ]].head(100).copy()
    pri["days_late_pred"]  = np.ceil(pri["days_late_pred"]).astype(int)
    pri["days_late_lower"] = np.ceil(pri["days_late_lower"]).astype(int)
    pri["days_late_upper"] = np.ceil(pri["days_late_upper"]).astype(int)
    pri["p_late"] = [
        round(_p_late(r["days_late_lower"], r["days_late_upper"]), 2)
        for _, r in pri.iterrows()
    ]
    pri["p_responds"] = [
        round(_p_responds(r["cust_std_days_late"]), 2)
        for _, r in pri.iterrows()
    ]
    pri = pri.drop(columns=["cust_std_days_late"])

    # --- Post-prediction KPIs ---
    call_count   = int((priority["action"] == "CALL").sum())
    remind_count = int((priority["action"] == "REMIND").sum())
    watch_count  = int((priority["action"] == "WATCH").sum())

    pi_widths = (predictions["days_late_upper"] - predictions["days_late_lower"]).abs()

    # Value at risk by tier ($ exposure in each bucket)
    tier_exposure = {}
    for tier in ["CALL", "REMIND", "WATCH", "OK"]:
        mask = priority["action"] == tier
        tier_exposure[tier.lower()] = round(
            float(priority.loc[mask, "total_open_amount"].sum()), 2
        )

    kpis = {
        "avg_days_late":   int(np.ceil(predictions["days_late_pred"].mean())),
        "total_at_risk":   int((predictions["days_late_pred"] > 0).sum()),
        "call_count":      call_count,
        "remind_count":    remind_count,
        "watch_count":     watch_count,
        "daily_recovery":  round(float(
            priority[priority["action"] == "CALL"]["expected_value"].sum()
        ), 2),
        "avg_pi_width":    round(float(pi_widths.mean()), 1),
        "value_at_risk_usd": round(float(
            predictions.loc[predictions["days_late_pred"] > 0, "total_open_amount"].sum()
        ), 2),
        "tier_exposure":   tier_exposure,
    }

    strategy_comparison = compute_strategy_comparison(priority)

    return jsonify({
        "invoices":            inv.to_dict("records"),
        "priority":            pri.to_dict("records"),
        "kpis":                kpis,
        "strategy_comparison": strategy_comparison,
    })


if __name__ == "__main__":
    app.run(debug=True)
