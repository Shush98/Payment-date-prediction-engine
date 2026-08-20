"""Local checks for the Phase 5 training logic. Pure pandas/numpy — no Spark, runs in seconds.

Run: python databricks/test_modeling.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "deploy"))

from modeling import (DEFAULT_KEYS, FEATURE_COLS, RAW_INPUT_COLS, PaymentLatenessModel,
                      apply_encoders, build_encoders, business_metrics, categorical_psi,
                      chronological_split, drift_label, expected_value,
                      late_classification_auc, model_metrics, p_late, p_responds,
                      population_stability_index)


def test_feature_cols_match_flask_app():
    """The Databricks model and the Flask app must agree on feature order, or a model
    exported from one scores garbage in the other."""
    from src.features import FEATURE_COLS as APP_COLS
    assert FEATURE_COLS == list(APP_COLS), f"drift:\n  databricks {FEATURE_COLS}\n  flask      {list(APP_COLS)}"


def test_decision_constants_match_flask_app():
    from src import decision
    assert (decision.CALL_COST, decision.DAYS_ACCELERATED,
            decision.DAILY_CAPITAL_RATE, decision.DAILY_CAPACITY) == (15.0, 3.0, 0.0003, 20)


def test_p_late_matches_flask_implementation():
    from src.decision import _p_late
    cases = [(-10.0, -2.0), (-2.0, 7.0), (6.0, 20.0), (0.0, 0.5), (2.0, 11.0)]
    mine = p_late([lo for lo, _ in cases], [hi for _, hi in cases])
    theirs = [_p_late(lo, hi) for lo, hi in cases]
    assert np.allclose(mine, theirs), f"{mine} != {theirs}"


def test_p_responds_matches_flask_implementation():
    from src.decision import _p_responds
    stds = [0.0, 3.0, 6.0, 16.0, 40.0]
    assert np.allclose(p_responds(stds), [_p_responds(s) for s in stds])


def test_expected_value_matches_flask_implementation():
    from src.decision import expected_value as app_ev
    rows = [(250_000.0, -2.0, 7.0, 6.0), (1_000.0, 1.0, 3.0, 2.0), (-50_000.0, 0.0, 9.0, 12.0)]
    mine = expected_value([r[0] for r in rows], [r[1] for r in rows],
                          [r[2] for r in rows], [r[3] for r in rows])
    theirs = [app_ev(*r) for r in rows]
    assert np.allclose(mine, theirs, atol=0.01), f"{mine} != {theirs}"


def test_chronological_split_has_no_overlap():
    df = pd.DataFrame({"posting_date": pd.date_range("2020-01-01", periods=100, freq="D"),
                       "v": range(100)}).sample(frac=1, random_state=0)   # shuffled input
    train, test = chronological_split(df, 0.2)
    assert len(train) == 80 and len(test) == 20
    assert train.posting_date.max() <= test.posting_date.min(), "folds overlap in time"


def test_encoders_fit_on_train_only():
    train = pd.DataFrame({"payment_terms": ["A", "B"], "business_code": ["U1", "U1"]})
    test = pd.DataFrame({"payment_terms": ["A", "ZZZ"], "business_code": ["U1", "U9"]})
    enc = build_encoders(train)
    out = apply_encoders(test, enc)
    assert out.payment_terms_encoded.tolist() == [0, -1], "unseen category must map to -1"
    assert out.business_code_encoded.tolist() == [0, -1]


def test_metrics_are_sane_on_perfect_prediction():
    y = np.array([0.0, 5.0, -3.0, 12.0])
    m = model_metrics(y, y, y - 1, y + 1)
    assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["bias"] == 0.0
    assert m["r2"] == 1.0
    assert m["within_3d"] == 100.0
    assert m["pi_coverage"] == 100.0


def test_auc_ranks_correctly():
    # Higher score for genuinely late invoices -> AUC 1.0
    y = np.array([-5.0, -1.0, 3.0, 10.0])
    assert late_classification_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert late_classification_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert late_classification_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5, "ties should give 0.5"
    assert np.isnan(late_classification_auc(np.array([1.0, 2.0]), [0.5, 0.6])), "one class -> nan"


def test_business_metrics_prefers_ev_ranking():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "invoice_amount": rng.uniform(500, 500_000, n),
        "cust_std_days_late": rng.uniform(0, 25, n),
    })
    lower = rng.uniform(-5, 2, n)
    upper = lower + rng.uniform(1, 15, n)
    b = business_metrics(df, lower, upper, capacity=20)
    # By construction the engine maximises the EV of the chosen 20.
    assert b["ev_decision_engine"] >= b["ev_by_amount"]
    assert b["ev_decision_engine"] >= b["ev_random"]
    assert b["ev_uplift_vs_by_amount"] == b["ev_decision_engine"] - b["ev_by_amount"]


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert population_stability_index(x, x) < 0.01
    # Same distribution, different sample -> still stable
    assert population_stability_index(x, rng.normal(0, 1, 5000)) < 0.1


def test_psi_detects_a_real_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    assert drift_label(population_stability_index(ref, rng.normal(0, 1, 5000))) == "stable"
    assert population_stability_index(ref, rng.normal(2.0, 1, 5000)) > 0.25
    assert drift_label(population_stability_index(ref, rng.normal(2.0, 1, 5000))) == "significant"


def test_psi_uses_reference_bins_not_current():
    """Binning on the current data would normalise away the very shift being measured."""
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 5000)
    shifted = rng.normal(0, 1, 5000) + 5.0
    assert population_stability_index(ref, shifted) > 1.0, "large shift must not be absorbed"


def test_psi_handles_degenerate_input():
    assert population_stability_index([], [1, 2, 3]) != population_stability_index([], [])  # both nan-safe
    assert np.isnan(population_stability_index([], [1, 2, 3]))
    assert population_stability_index([5.0] * 100, [5.0] * 100) == 0.0, "constant reference"


def test_categorical_psi_flags_new_and_vanishing_categories():
    ref = ["A"] * 60 + ["B"] * 40
    assert categorical_psi(ref, ["A"] * 60 + ["B"] * 40) < 0.01
    # A category that never appeared in training shows up as a third of traffic
    assert categorical_psi(ref, ["A"] * 40 + ["B"] * 27 + ["C"] * 33) > 0.25
    # An expected category disappears entirely
    assert categorical_psi(ref, ["A"] * 100) > 0.25


def test_drift_label_thresholds():
    assert drift_label(0.05) == "stable"
    assert drift_label(0.15) == "moderate"
    assert drift_label(0.9) == "significant"
    assert drift_label(float("nan")) == "unknown"


class _Stub:
    def __init__(self, v): self.v = v
    def predict(self, X): return np.full(len(X), self.v, dtype=float)


ENCODERS = {"payment_terms": {"NAA8": 0, "NAH4": 1}, "business_code": {"U001": 0}}


def _raw_row(terms="NAA8", code="U001"):
    row = {c: 0.0 for c in RAW_INPUT_COLS if c not in ("payment_terms", "business_code")}
    row["payment_terms"] = terms
    row["business_code"] = code
    return row


def test_wrapper_orders_crossed_quantiles():
    # lower model returns a HIGHER value than upper -> must be swapped, not passed through
    m = PaymentLatenessModel(_Stub(2.0), _Stub(9.0), _Stub(1.0), ENCODERS)
    out = m.predict(pd.DataFrame([_raw_row()]))
    assert out.days_late_lower.iloc[0] == 1.0
    assert out.days_late_upper.iloc[0] == 9.0
    assert (out.days_late_lower <= out.days_late_upper).all()
    assert 0.0 <= out.p_late.iloc[0] <= 1.0


def test_wrapper_preserves_index():
    m = PaymentLatenessModel(_Stub(1.0), _Stub(0.0), _Stub(2.0), ENCODERS)
    df = pd.DataFrame([_raw_row() for _ in range(3)], index=[10, 20, 30])
    assert m.predict(df).index.tolist() == [10, 20, 30]


def test_wrapper_encodes_internally():
    """The model owns its category mapping. Callers pass raw strings; an unseen
    category must degrade to -1 rather than raise."""
    captured = {}

    class _Capture:
        def predict(self, X):
            captured["X"] = X
            return np.zeros(len(X))

    m = PaymentLatenessModel(_Capture(), _Stub(0.0), _Stub(1.0), ENCODERS)
    m.predict(pd.DataFrame([_raw_row("NAH4"), _raw_row("UNSEEN")]))

    col = FEATURE_COLS.index("payment_terms_encoded")
    assert captured["X"][0, col] == 1, "known category encoded wrongly"
    assert captured["X"][1, col] == -1, "unseen category must map to -1, not raise"


def test_raw_input_cols_cover_every_feature():
    """Anything the model needs must be derivable from what callers are asked for."""
    derived = {"payment_terms_encoded", "business_code_encoded"}
    assert set(FEATURE_COLS) - derived <= set(RAW_INPUT_COLS)
    assert "payment_terms" in RAW_INPUT_COLS and "business_code" in RAW_INPUT_COLS
    assert not [c for c in RAW_INPUT_COLS if c.endswith("_encoded")], "raw input must not be pre-encoded"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {str(e).splitlines()[0][:110]}")
    print("\nAll modeling checks passed." if not failed else f"\n{failed} FAILED")
    sys.exit(1 if failed else 0)
