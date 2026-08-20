"""Training-side logic for Phase 5. Pure pandas/numpy/sklearn — no Spark, no Databricks.

Kept separate from transforms.py so it can be tested without a Spark session, and so
the notebook stays orchestration only.

Feature order is identical to deploy/src/features.py FEATURE_COLS. A model trained here
must be scoreable by the Flask app and vice versa, so the two lists cannot drift.
"""

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

# What the registered model expects as input: the numeric features plus the two
# categoricals as RAW strings. The model encodes them itself, so callers never have to
# reproduce the mapping.
RAW_INPUT_COLS = [c for c in FEATURE_COLS if not c.endswith("_encoded")] + \
                 ["payment_terms", "business_code"]

# Cold-start fallbacks are fitted on the train fold and must be reused at inference, or
# customers with no history get different values than the model was trained with. They
# are logged as MLflow params (default_*) so they travel with the model version.
DEFAULT_KEYS = ["cust_avg_days_late", "cust_std_days_late", "cust_invoice_count",
                "cust_min_late", "cust_max_late", "cust_avg_amount"]

# Mirrors deploy/src/decision.py. Duplicated rather than imported so this module has no
# dependency on the Flask app's layout; test_modeling.py asserts the two stay in sync.
CALL_COST = 15.0
DAYS_ACCELERATED = 3.0
DAILY_CAPITAL_RATE = 0.0003
DAILY_CAPACITY = 20


def build_encoders(train: pd.DataFrame) -> dict:
    """Category → integer code, fit on the train fold only.

    Unseen categories at scoring time map to -1 rather than raising, which is what a
    genuinely new payment-terms code should do.
    """
    return {
        "payment_terms": {v: i for i, v in enumerate(sorted(train["payment_terms"].dropna().unique()))},
        "business_code": {v: i for i, v in enumerate(sorted(train["business_code"].dropna().unique()))},
    }


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    out = df.copy()
    out["payment_terms_encoded"] = out["payment_terms"].map(encoders["payment_terms"]).fillna(-1).astype(int)
    out["business_code_encoded"] = out["business_code"].map(encoders["business_code"]).fillna(-1).astype(int)
    return out


def chronological_split(df: pd.DataFrame, test_fraction=0.2):
    """Train on the past, test on the future. The open book is always later than
    everything trained on, so a shuffled split would measure the wrong thing."""
    ordered = df.sort_values("posting_date").reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def p_late(lower, upper):
    """Probability of paying late, from the prediction interval. Mirrors
    deploy/src/decision.py::_p_late."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = np.maximum(upper - lower, 1.0)
    out = np.clip(upper / width, 0.0, 1.0)
    out = np.where(upper < 0, 0.05, out)
    return np.where(lower > 5, 0.95, out)


def p_responds(cust_std):
    return np.clip(1.0 - np.asarray(cust_std, dtype=float) / 20.0, 0.2, 0.8)


def expected_value(amount, lower, upper, cust_std):
    benefit = (p_late(lower, upper) * p_responds(cust_std)
               * DAYS_ACCELERATED * DAILY_CAPITAL_RATE * np.abs(np.asarray(amount, dtype=float)))
    return benefit - CALL_COST


def model_metrics(y, pred, lower, upper) -> dict:
    """Statistical quality. Interval coverage matters as much as MAE here, because the
    decision engine derives P(late) from the interval rather than from the point estimate."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    err = y - pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_ae": float(np.median(np.abs(err))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot else float("nan"),
        "bias": float(np.mean(pred - y)),
        "within_1d": float(np.mean(np.abs(err) <= 1) * 100),
        "within_3d": float(np.mean(np.abs(err) <= 3) * 100),
        "within_7d": float(np.mean(np.abs(err) <= 7) * 100),
        "pi_coverage": float(np.mean((y >= lower) & (y <= upper)) * 100),
        "pi_mean_width": float(np.mean(upper - lower)),
        "late_auc": late_classification_auc(y, p_late(lower, upper)),
    }


def late_classification_auc(y_true_days, score) -> float:
    """AUC for 'will this be late' derived from the interval. Rank-based, no sklearn
    dependency, and returns nan when only one class is present."""
    label = (np.asarray(y_true_days, float) > 0).astype(int)
    n_pos, n_neg = int(label.sum()), int((1 - label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(np.asarray(score, float), kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ranks within ties so tied scores don't inflate the statistic.
    s = np.asarray(score, float)[order]
    i = 0
    r = ranks[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks[order] = r
    return float((ranks[label == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def business_metrics(df: pd.DataFrame, lower, upper, capacity=DAILY_CAPACITY) -> dict:
    """Value of the ranking, not the accuracy of the prediction.

    Compares the engine's top-N by expected value against the two strategies a
    collections team would otherwise use. Self-consistent under the engine's own
    assumptions - it compares orderings, and is NOT a measured business outcome.
    """
    ev = expected_value(df["invoice_amount"], lower, upper, df["cust_std_days_late"])
    work = pd.DataFrame({"ev": ev, "amount": df["invoice_amount"].to_numpy()})

    rng = np.random.default_rng(42)
    random_ev = float(work.iloc[rng.choice(len(work), min(capacity, len(work)), replace=False)].ev.sum())
    by_amount_ev = float(work.nlargest(capacity, "amount").ev.sum())
    engine_ev = float(work.nlargest(capacity, "ev").ev.sum())

    return {
        "ev_random": random_ev,
        "ev_by_amount": by_amount_ev,
        "ev_decision_engine": engine_ev,
        "ev_uplift_vs_by_amount": engine_ev - by_amount_ev,
        "positive_ev_invoices": int((work.ev > 0).sum()),
    }


def population_stability_index(reference, current, bins=10) -> float:
    """PSI between a reference and a current distribution.

    Bin edges come from the REFERENCE (training) quantiles, so the question asked is
    "where does today's traffic fall relative to what the model was fitted on".
    Re-binning on the current data would hide exactly the shift being looked for.

    Convention: < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # near-constant reference, nothing to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    eps = 1e-6
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), eps, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def categorical_psi(reference, current) -> float:
    """PSI over category shares. Categories absent from one side get a floor rather
    than being dropped, so a category appearing for the first time still registers."""
    r = pd.Series(list(reference)).value_counts(normalize=True)
    c = pd.Series(list(current)).value_counts(normalize=True)
    if r.empty or c.empty:
        return float("nan")
    idx = r.index.union(c.index)
    eps = 1e-6
    r = r.reindex(idx).fillna(0.0).clip(lower=eps)
    c = c.reindex(idx).fillna(0.0).clip(lower=eps)
    return float(((c - r) * np.log(c / r)).sum())


def drift_label(psi) -> str:
    if not np.isfinite(psi):
        return "unknown"
    if psi < 0.1:
        return "stable"
    return "moderate" if psi < 0.25 else "significant"


class PaymentLatenessModel:
    """Bundles the three quantile models *and their encoders* behind one predict() call.

    Registered to Unity Catalog as a single mlflow.pyfunc model. Two reasons it owns
    the encoding rather than expecting pre-encoded input:

    - The decision engine needs the interval as well as the point estimate, so three
      separately registered models would push reassembly onto every caller.
    - A model that requires callers to reproduce its category->code mapping is a
      train/serve skew waiting to happen. Inference passes raw `payment_terms` and
      `business_code`; the model applies the mapping it was fitted with.

    Wrapped as an mlflow.pyfunc.PythonModel in the notebook; kept framework-free here
    so it can be tested without mlflow installed.
    """

    def __init__(self, model_mean, model_lower, model_upper, encoders, feature_cols=FEATURE_COLS):
        self.model_mean = model_mean
        self.model_lower = model_lower
        self.model_upper = model_upper
        self.encoders = encoders
        self.feature_cols = list(feature_cols)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = apply_encoders(df, self.encoders)[self.feature_cols].to_numpy()
        lower = self.model_lower.predict(X)
        upper = self.model_upper.predict(X)
        # A quantile pair can cross on hard rows; an inverted interval would make
        # p_late meaningless downstream, so order them.
        lo, hi = np.minimum(lower, upper), np.maximum(lower, upper)
        return pd.DataFrame({
            "days_late_pred": self.model_mean.predict(X),
            "days_late_lower": lo,
            "days_late_upper": hi,
            "p_late": p_late(lo, hi),
        }, index=df.index)
