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

from modeling import (FEATURE_COLS, PaymentLatenessModel, apply_encoders, build_encoders,
                      business_metrics, chronological_split, expected_value,
                      late_classification_auc, model_metrics, p_late, p_responds)


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


class _Stub:
    def __init__(self, v): self.v = v
    def predict(self, X): return np.full(len(X), self.v, dtype=float)


def test_wrapper_orders_crossed_quantiles():
    # lower model returns a HIGHER value than upper -> must be swapped, not passed through
    m = PaymentLatenessModel(_Stub(2.0), _Stub(9.0), _Stub(1.0))
    df = pd.DataFrame([{c: 0.0 for c in FEATURE_COLS}])
    out = m.predict(df)
    assert out.days_late_lower.iloc[0] == 1.0
    assert out.days_late_upper.iloc[0] == 9.0
    assert (out.days_late_lower <= out.days_late_upper).all()
    assert 0.0 <= out.p_late.iloc[0] <= 1.0


def test_wrapper_preserves_index():
    m = PaymentLatenessModel(_Stub(1.0), _Stub(0.0), _Stub(2.0))
    df = pd.DataFrame([{c: 0.0 for c in FEATURE_COLS} for _ in range(3)], index=[10, 20, 30])
    assert m.predict(df).index.tolist() == [10, 20, 30]


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
