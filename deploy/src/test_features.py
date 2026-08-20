"""Point-in-time correctness check for the customer-history join.

Run: python -m src.test_features      (from the deploy/ directory)
"""

import numpy as np
import pandas as pd

from src.features import add_features, build_customer_timeline, HIST_COLS

ENCODERS = {"payment_terms": {"NAA8": 0}, "business_code": {"U001": 0}}
DEFAULTS = {c: 0.0 for c in HIST_COLS}


def _invoice(cust, posting, clear, days_late, amount=1000.0):
    return dict(cust_number=cust, posting_date=pd.Timestamp(posting),
                clear_date=pd.Timestamp(clear), days_late=days_late,
                total_open_amount=amount, cust_payment_terms="NAA8",
                business_code="U001")


def test_history_excludes_future_and_self():
    # Customer A: invoice 1 clears 90 days late on Feb 1, invoice 2 is raised Mar 1.
    closed = pd.DataFrame([
        _invoice("A", "2020-01-01", "2020-02-01", 90),
        _invoice("A", "2020-03-01", "2020-04-01", 2),
    ])
    out = add_features(closed, ENCODERS, build_customer_timeline(closed), DEFAULTS)

    # Row 1 had no cleared history when it was raised -> defaults, not its own 90.
    assert out.loc[0, "cust_invoice_count"] == 0, "first invoice must not see itself"
    assert out.loc[0, "cust_avg_days_late"] == 0.0

    # Row 2 sees only invoice 1's outcome (90), never its own (2).
    assert out.loc[1, "cust_invoice_count"] == 1
    assert out.loc[1, "cust_avg_days_late"] == 90.0, "must not average in its own target"


def test_same_day_clear_is_not_visible():
    # An invoice raised the same day another clears must not see that outcome.
    closed = pd.DataFrame([
        _invoice("B", "2020-01-01", "2020-03-01", 5),
        _invoice("B", "2020-03-01", "2020-05-01", 7),
    ])
    out = add_features(closed, ENCODERS, build_customer_timeline(closed), DEFAULTS)
    assert out.loc[1, "cust_invoice_count"] == 0, "same-day clear must be excluded"


def test_row_order_is_preserved():
    closed = pd.DataFrame([
        _invoice("C", "2020-06-01", "2020-07-01", 1),
        _invoice("D", "2020-01-01", "2020-02-01", 2),
        _invoice("C", "2020-09-01", "2020-10-01", 3),
    ])
    out = add_features(closed, ENCODERS, build_customer_timeline(closed), DEFAULTS)
    assert list(out["cust_number"]) == ["C", "D", "C"], "as-of join must not reorder rows"
    assert np.isclose(out.loc[2, "cust_avg_days_late"], 1.0)


def test_customers_are_not_mixed():
    closed = pd.DataFrame([
        _invoice("E", "2020-01-01", "2020-02-01", 50),
        _invoice("F", "2020-03-01", "2020-04-01", 3),
    ])
    out = add_features(closed, ENCODERS, build_customer_timeline(closed), DEFAULTS)
    assert out.loc[1, "cust_invoice_count"] == 0, "F must not inherit E's history"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nAll point-in-time checks passed.")
