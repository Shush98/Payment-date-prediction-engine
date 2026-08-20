"""Local PySpark checks for the medallion transforms.

Runs on plain pyspark - no Databricks runtime needed - so the point-in-time rule
can be verified before anything is deployed. The equivalent pandas checks live in
deploy/src/test_features.py; both must agree, because the Spark pipeline and the
Flask app are supposed to compute the same features.

Run: python databricks/test_transforms.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from transforms import (HIST_COLS, asof_join_history, build_customer_timeline,
                        build_gold_customer_behavior, dedupe_events, join_outcomes,
                        validate_invoices)

DEFAULTS = {c: 0.0 for c in HIST_COLS}


def spark_session():
    return (SparkSession.builder
            .appName("transform-tests")
            .master("local[2]")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.ui.enabled", "false")
            # Databricks serverless runs ANSI on; OSS pyspark defaults it off. Without
            # this the local suite is a weaker test than production and silently misses
            # cast failures - which is exactly how the to_date() crash reached the
            # workspace despite green tests.
            .config("spark.sql.ansi.enabled", "true")
            .getOrCreate())


def _outcomes(spark, rows):
    # (customer_id, invoice_id, posting, due, clear, amount)
    return spark.createDataFrame(
        rows, "customer_id string, invoice_id string, posting_date date, "
              "due_date date, clear_date date, invoice_amount double"
    ).withColumn("days_late", F.datediff("clear_date", "due_date"))


def test_timeline_is_cumulative(spark):
    out = _outcomes(spark, [
        ("A", "i1", date(2020, 1, 1), date(2020, 1, 31), date(2020, 4, 30), 100.0),  # 90 late
        ("A", "i2", date(2020, 3, 1), date(2020, 3, 31), date(2020, 4, 2), 100.0),   # 2 late
    ])
    tl = {r["clear_date"]: r for r in build_customer_timeline(out).collect()}
    assert tl[date(2020, 4, 2)]["cust_invoice_count"] == 1
    assert tl[date(2020, 4, 2)]["cust_avg_days_late"] == 2.0
    assert tl[date(2020, 4, 30)]["cust_invoice_count"] == 2
    assert tl[date(2020, 4, 30)]["cust_avg_days_late"] == 46.0   # (90 + 2) / 2


def test_same_day_clears_collapse_to_one_row(spark):
    """(customer_id, clear_date) must be unique - it is the feature table's primary key.
    Several invoices clearing on the same day previously produced one row each, which
    UC rejects and which made the as-of join's tie-break arbitrary."""
    out = _outcomes(spark, [
        ("A", "i1", date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1), 100.0),   # 29 late
        ("A", "i2", date(2020, 1, 5), date(2020, 2, 25), date(2020, 3, 1), 300.0),  # 5 late, same day
        ("A", "i3", date(2020, 1, 9), date(2020, 4, 1), date(2020, 4, 3), 200.0),   # 2 late
    ])
    tl = build_customer_timeline(out)

    keys = [(r["customer_id"], r["clear_date"]) for r in tl.collect()]
    assert len(keys) == len(set(keys)), f"non-unique primary key: {keys}"
    assert len(keys) == 2, f"expected one row per clear_date, got {len(keys)}"

    # The surviving 2020-03-01 row must reflect BOTH of that day's payments.
    day1 = [r for r in tl.collect() if r["clear_date"] == date(2020, 3, 1)][0]
    assert day1["cust_invoice_count"] == 2.0, day1["cust_invoice_count"]
    assert day1["cust_avg_days_late"] == 17.0, day1["cust_avg_days_late"]   # (29 + 5) / 2
    assert day1["cust_max_late"] == 29.0


def test_asof_uses_fully_settled_same_day_state(spark):
    """A later invoice must see all of a prior day's payments, not just the first."""
    out = _outcomes(spark, [
        ("A", "i1", date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1), 100.0),
        ("A", "i2", date(2020, 1, 5), date(2020, 2, 25), date(2020, 3, 1), 300.0),
        ("A", "i3", date(2020, 6, 1), date(2020, 7, 1), date(2020, 7, 2), 200.0),
    ])
    joined = asof_join_history(out, build_customer_timeline(out), DEFAULTS)
    i3 = [r for r in joined.collect() if r["invoice_id"] == "i3"][0]
    assert i3["cust_invoice_count"] == 2.0, "undercounted same-day history"
    assert i3["cust_avg_days_late"] == 17.0


def test_invoice_never_sees_its_own_outcome(spark):
    out = _outcomes(spark, [
        ("A", "i1", date(2020, 1, 1), date(2020, 1, 31), date(2020, 4, 30), 100.0),
        ("A", "i2", date(2020, 5, 1), date(2020, 5, 31), date(2020, 6, 2), 100.0),
    ])
    joined = asof_join_history(out, build_customer_timeline(out), DEFAULTS)
    rows = {r["invoice_id"]: r for r in joined.collect()}

    # i1 was raised before anything of A's had cleared -> defaults, not its own 90.
    assert rows["i1"]["cust_invoice_count"] == 0.0
    assert rows["i1"]["cust_avg_days_late"] == 0.0
    # i2 sees only i1's outcome (90), never its own (2).
    assert rows["i2"]["cust_invoice_count"] == 1.0
    assert rows["i2"]["cust_avg_days_late"] == 90.0


def test_same_day_clear_is_invisible(spark):
    # A payment clearing on the same day an invoice is raised must not be visible.
    out = _outcomes(spark, [
        ("B", "i1", date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1), 100.0),
        ("B", "i2", date(2020, 3, 1), date(2020, 4, 1), date(2020, 5, 1), 100.0),
    ])
    joined = asof_join_history(out, build_customer_timeline(out), DEFAULTS)
    rows = {r["invoice_id"]: r for r in joined.collect()}
    assert rows["i2"]["cust_invoice_count"] == 0.0, "same-day clear leaked into features"


def test_customers_are_not_mixed(spark):
    out = _outcomes(spark, [
        ("E", "i1", date(2020, 1, 1), date(2020, 1, 31), date(2020, 2, 1), 100.0),
        ("F", "i2", date(2020, 3, 1), date(2020, 3, 31), date(2020, 4, 1), 100.0),
    ])
    joined = asof_join_history(out, build_customer_timeline(out), DEFAULTS)
    rows = {r["invoice_id"]: r for r in joined.collect()}
    assert rows["i2"]["cust_invoice_count"] == 0.0, "F inherited E's history"


def test_one_row_per_invoice(spark):
    # The as-of join fans out before row_number picks the latest; make sure it collapses.
    out = _outcomes(spark, [
        ("G", "i1", date(2020, 1, 1), date(2020, 1, 31), date(2020, 2, 1), 100.0),
        ("G", "i2", date(2020, 2, 5), date(2020, 3, 1), date(2020, 3, 5), 100.0),
        ("G", "i3", date(2020, 4, 1), date(2020, 5, 1), date(2020, 5, 5), 100.0),
    ])
    joined = asof_join_history(out, build_customer_timeline(out), DEFAULTS)
    assert joined.count() == 3, "as-of join duplicated invoices"
    i3 = [r for r in joined.collect() if r["invoice_id"] == "i3"][0]
    assert i3["cust_invoice_count"] == 2.0, "i3 should see i1 and i2 only"


def test_gold_recency_windows(spark):
    """Regression: recency was computed with max().over(Window) inside avg(), which
    Spark rejects ('window function inside an aggregate') and which also forces every
    row through one partition."""
    out = _outcomes(spark, [
        ("A", "i1", date(2019, 12, 1), date(2019, 12, 22), date(2020, 1, 1), 100.0),   # 10 late, old
        ("A", "i2", date(2020, 5, 1), date(2020, 5, 30), date(2020, 6, 1), 200.0),     # 2 late, recent
        ("A", "i3", date(2020, 5, 20), date(2020, 6, 11), date(2020, 6, 15), 300.0),   # 4 late, recent
    ])
    g = build_gold_customer_behavior(out).collect()[0]      # as_of resolves to 2020-06-15

    assert g["invoice_count"] == 3
    assert abs(g["avg_days_late"] - 16.0 / 3) < 1e-6, g["avg_days_late"]
    # 90-day window starts 2020-03-17, so only i2 and i3 count: (2 + 4) / 2
    assert g["recent_90d_avg_days_late"] == 3.0, g["recent_90d_avg_days_late"]
    # 30-day window starts 2020-05-16: i2 and i3, both late
    assert g["recent_30d_late_rate"] == 1.0, g["recent_30d_late_rate"]
    assert g["as_of_date"] == date(2020, 6, 15)


def test_gold_respects_explicit_as_of(spark):
    out = _outcomes(spark, [
        ("A", "i1", date(2019, 12, 1), date(2019, 12, 22), date(2020, 1, 1), 100.0),
        ("A", "i2", date(2020, 5, 1), date(2020, 5, 30), date(2020, 6, 1), 200.0),
    ])
    # Anchored far in the future, nothing falls inside the 90-day window.
    g = build_gold_customer_behavior(out, as_of=date(2021, 1, 1)).collect()[0]
    assert g["recent_90d_avg_days_late"] is None
    assert g["invoice_count"] == 2, "lifetime aggregates must be unaffected by as_of"


def test_dedupe_keeps_latest_ingestion(spark):
    df = spark.createDataFrame(
        [("e1", "a", 1), ("e1", "b", 2), ("e2", "c", 1)],
        "_event_id string, payload string, _ingested_at int")
    got = {r["_event_id"]: r["payload"] for r in dedupe_events(df).collect()}
    assert got == {"e1": "b", "e2": "c"}


def test_validation_quarantines_bad_rows(spark):
    df = spark.createDataFrame([
        ("e1", "invoice_created", "i1", "c1", 100.0, "2020-01-01", "2020-01-31"),
        ("e2", "invoice_created", "i2", "c1", 100.0, "2020-05-01", "2020-01-01"),  # due < posting
        ("e3", "invoice_created", "i3", "c1", None, "2020-01-01", "2020-01-31"),   # null amount
        ("e4", "invoice_created", "i4", "c1", 100.0, "not-a-date", "2020-01-31"),  # bad date
    ], "_event_id string, event_type string, invoice_id string, customer_id string, "
       "invoice_amount double, posting_date string, due_date string")
    clean, bad = validate_invoices(df)
    assert clean.count() == 1
    reasons = {r["_reject_reason"] for r in bad.collect()}
    assert reasons == {"due_before_posting", "null_amount", "unparseable_posting_date"}, reasons


def test_malformed_dates_quarantine_instead_of_throwing(spark):
    """Regression: under ANSI mode to_date() raises CAST_INVALID_INPUT rather than
    returning NULL, which took down the whole Silver notebook on the first corrupt
    row. These are the exact values the generator injects."""
    df = spark.createDataFrame([
        ("e1", "invoice_created", "i1", "c1", 100.0, "31/02/2020", "2020-03-01"),   # wrong format
        ("e2", "invoice_created", "i2", "c1", 100.0, "2020-02-31", "2020-03-01"),   # no such day
        ("e3", "invoice_created", "i3", "c1", 100.0, "not-a-date", "2020-03-01"),
        ("e4", "invoice_created", "i4", "c1", 100.0, "2020-01-01", "garbage"),
        ("e5", "invoice_created", "i5", "c1", 100.0, "2020-01-01", "2020-01-31"),   # good
    ], "_event_id string, event_type string, invoice_id string, customer_id string, "
       "invoice_amount double, posting_date string, due_date string")

    clean, bad = validate_invoices(df)          # must not raise
    assert clean.count() == 1, "only the well-formed row should survive"
    got = {r["invoice_id"]: r["_reject_reason"] for r in bad.collect()}
    assert got == {"i1": "unparseable_posting_date",
                   "i2": "unparseable_posting_date",
                   "i3": "unparseable_posting_date",
                   "i4": "unparseable_due_date"}, got


def test_join_outcomes_keeps_open_invoices(spark):
    inv = spark.createDataFrame(
        [("i1", "c1", date(2020, 1, 1), date(2020, 1, 31), 100.0),
         ("i2", "c1", date(2020, 2, 1), date(2020, 2, 28), 200.0)],
        "invoice_id string, customer_id string, posting_date date, due_date date, invoice_amount double")
    pay = spark.createDataFrame([("i1", date(2020, 2, 5))], "invoice_id string, clear_date date")
    out = {r["invoice_id"]: r for r in join_outcomes(inv, pay).collect()}
    assert out["i1"]["days_late"] == 5 and out["i1"]["is_open"] == 0
    assert out["i2"]["days_late"] is None and out["i2"]["is_open"] == 1


if __name__ == "__main__":
    spark = spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(spark)
                print(f"ok    {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:
                # A raised Spark error is a failure too - notably ANSI cast errors,
                # which are the whole point of the malformed-date test.
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {str(e).splitlines()[0][:110]}")
    spark.stop()
    print("\nAll point-in-time checks passed." if not failed else f"\n{failed} FAILED")
    sys.exit(1 if failed else 0)
