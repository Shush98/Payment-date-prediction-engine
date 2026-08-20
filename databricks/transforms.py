"""Pure PySpark transformations for the medallion pipeline.

Kept out of the notebooks so the logic can be tested locally with plain pyspark
(see test_transforms.py) instead of only inside a Databricks workspace. The
notebooks import this module and handle I/O, streaming and table registration.

The point-in-time rule implemented in build_customer_timeline / asof_join_history
is the Spark port of deploy/src/features.py. It must stay equivalent: a prediction
may only see payment outcomes that had already cleared strictly before the invoice
was raised.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, StringType, DoubleType,
                               TimestampType, DateType)

# --- Event contract -----------------------------------------------------------
# Two event types, because payment outcome is a delayed label: an invoice is
# raised on posting_date, and the matching payment arrives days or weeks later.
# Modelling them as separate events is what makes the delayed-label join real
# rather than a reshuffle of a static CSV.

INVOICE_EVENT_SCHEMA = StructType([
    StructField("_event_id", StringType(), False),
    StructField("_source", StringType(), True),
    StructField("event_type", StringType(), False),      # invoice_created | invoice_paid
    StructField("event_timestamp", StringType(), True),
    StructField("invoice_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("customer_name", StringType(), True),
    StructField("invoice_amount", DoubleType(), True),
    StructField("posting_date", StringType(), True),
    StructField("due_date", StringType(), True),
    StructField("payment_terms", StringType(), True),
    StructField("business_code", StringType(), True),
    StructField("clear_date", StringType(), True),       # populated on invoice_paid only
])

HIST_COLS = ["cust_avg_days_late", "cust_std_days_late", "cust_invoice_count",
             "cust_min_late", "cust_max_late", "cust_avg_amount"]

MAX_PLAUSIBLE_AMOUNT = 100_000_000.0


def safe_date(col):
    """Parse to date, returning NULL on malformed input instead of raising.

    Databricks serverless runs with spark.sql.ansi.enabled=true, where to_date()
    THROWS on a bad string rather than returning NULL - which would kill the whole
    pipeline on the first corrupt row, exactly the rows quarantine exists to catch.

    try_to_date() is Databricks-only, so try_to_timestamp().cast(date) is used
    instead: it is in OSS pyspark 3.5 as well, keeping the local tests faithful.
    It also returns NULL for calendar-invalid dates like 2020-02-31.
    """
    return F.try_to_timestamp(col).cast("date")


# --- Silver -------------------------------------------------------------------

def dedupe_events(bronze: DataFrame) -> DataFrame:
    """One row per _event_id, keeping the latest ingestion.

    The generator can replay, and Auto Loader guarantees at-least-once, so the
    same event can land twice. Dedupe on the producer-assigned id rather than on
    the whole row, which would let a re-send with a new _ingested_at slip through.
    """
    w = Window.partitionBy("_event_id").orderBy(F.col("_ingested_at").desc())
    return (bronze
            .withColumn("_rn", F.row_number().over(w))
            .filter(F.col("_rn") == 1)
            .drop("_rn"))


def validate_invoices(events: DataFrame):
    """Split invoice_created events into (clean, quarantined) with a failure reason.

    Bad rows are quarantined rather than dropped so the data-quality table can
    report on them and they stay inspectable.
    """
    inv = (events.filter(F.col("event_type") == "invoice_created")
           .withColumn("posting_date", safe_date("posting_date"))
           .withColumn("due_date", safe_date("due_date")))

    reason = (F.when(F.col("invoice_id").isNull(), "null_invoice_id")
              .when(F.col("customer_id").isNull(), "null_customer_id")
              .when(F.col("posting_date").isNull(), "unparseable_posting_date")
              .when(F.col("due_date").isNull(), "unparseable_due_date")
              .when(F.col("due_date") < F.col("posting_date"), "due_before_posting")
              .when(F.col("invoice_amount").isNull(), "null_amount")
              .when(F.abs(F.col("invoice_amount")) > MAX_PLAUSIBLE_AMOUNT, "implausible_amount")
              .otherwise(None))

    tagged = inv.withColumn("_reject_reason", reason)
    clean = tagged.filter(F.col("_reject_reason").isNull()).drop("_reject_reason")
    quarantine = tagged.filter(F.col("_reject_reason").isNotNull())
    return clean, quarantine


def validate_payments(events: DataFrame):
    pay = (events.filter(F.col("event_type") == "invoice_paid")
           .withColumn("clear_date", safe_date("clear_date")))
    reason = (F.when(F.col("invoice_id").isNull(), "null_invoice_id")
              .when(F.col("clear_date").isNull(), "unparseable_clear_date")
              .otherwise(None))
    tagged = pay.withColumn("_reject_reason", reason)
    return (tagged.filter(F.col("_reject_reason").isNull()).drop("_reject_reason"),
            tagged.filter(F.col("_reject_reason").isNotNull()))


def data_quality_metrics(spark, stage: str, rows_in: int,
                         clean: DataFrame, quarantine: DataFrame) -> DataFrame:
    """One row per pipeline run per stage, for the monitoring dashboard.

    Counts are passed/computed once by the caller; recounting a streaming-derived
    DataFrame re-executes its whole plan.
    """
    by_reason = {r["_reject_reason"]: r["n"] for r in
                 quarantine.groupBy("_reject_reason").agg(F.count("*").alias("n")).collect()}
    n_clean = clean.count()
    n_bad = sum(by_reason.values())
    return spark.createDataFrame(
        [(stage, int(rows_in), int(n_clean), int(n_bad),
          float(n_bad) / rows_in if rows_in else 0.0,
          str(by_reason))],
        "stage string, rows_in bigint, rows_clean bigint, rows_quarantined bigint, "
        "reject_rate double, reject_reasons string",
    ).withColumn("run_timestamp", F.current_timestamp())


def join_outcomes(invoices: DataFrame, payments: DataFrame) -> DataFrame:
    """Attach payment outcome to invoices. Unpaid invoices survive with null clear_date.

    days_late is the label: negative means paid early. Only rows where the payment
    has actually arrived carry a label - the rest are the open book to predict.
    """
    pay = payments.select("invoice_id", "clear_date")
    return (invoices.join(pay, on="invoice_id", how="left")
            .withColumn("days_late", F.datediff("clear_date", "due_date"))
            .withColumn("is_open", F.col("clear_date").isNull().cast("int")))


# --- Gold ---------------------------------------------------------------------

def build_gold_customer_behavior(outcomes: DataFrame, as_of=None) -> DataFrame:
    """Descriptive per-customer aggregates for dashboards and segmentation.

    NOT model features - this is the full-history view by construction and would
    leak if fed to training. The model uses build_customer_timeline instead.

    `as_of` anchors the recency windows. It is resolved to a scalar first rather
    than computed with max().over(): Spark forbids a window function inside an
    aggregate, and an unpartitioned window would also funnel every row through a
    single partition.
    """
    paid = outcomes.filter(F.col("clear_date").isNotNull())
    if as_of is None:
        row = paid.agg(F.max("clear_date").alias("m")).first()
        as_of = row["m"] if row is not None else None

    def within(days, value):
        if as_of is None:                     # no cleared invoices yet
            return F.lit(None).cast("double")
        return F.when(F.col("clear_date") >= F.date_sub(F.lit(as_of), days), value)

    return (paid.groupBy("customer_id")
            .agg(F.avg("days_late").alias("avg_days_late"),
                 F.stddev("days_late").alias("std_days_late"),
                 F.expr("percentile_approx(days_late, 0.5)").alias("median_days_late"),
                 F.avg((F.col("days_late") > 0).cast("double")).alias("late_rate"),
                 F.count("*").alias("invoice_count"),
                 F.avg("invoice_amount").alias("avg_invoice_amount"),
                 F.max("days_late").alias("max_days_late"),
                 F.avg(within(90, F.col("days_late"))).alias("recent_90d_avg_days_late"),
                 F.avg(within(30, (F.col("days_late") > 0).cast("double"))).alias("recent_30d_late_rate"))
            .withColumn("as_of_date", F.lit(as_of)))


# --- Feature table (point-in-time) --------------------------------------------

def build_customer_timeline(outcomes: DataFrame) -> DataFrame:
    """Running per-customer aggregates stamped with clear_date.

    Each row is what was knowable about a customer the instant their nth invoice
    cleared. Registered as a Unity Catalog time-series feature table keyed on
    (customer_id, clear_date) so lookups are point-in-time by construction.

    Spark port of deploy/src/features.build_customer_timeline.
    """
    paid = outcomes.filter(F.col("clear_date").isNotNull())
    w = (Window.partitionBy("customer_id").orderBy("clear_date")
         .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    running = (paid
               .select("customer_id", "clear_date", "days_late", "invoice_amount")
               .withColumn("cust_avg_days_late", F.avg("days_late").over(w))
               .withColumn("cust_std_days_late", F.coalesce(F.stddev("days_late").over(w), F.lit(0.0)))
               .withColumn("cust_invoice_count", F.count("days_late").over(w).cast("double"))
               .withColumn("cust_min_late", F.min("days_late").over(w).cast("double"))
               .withColumn("cust_max_late", F.max("days_late").over(w).cast("double"))
               .withColumn("cust_avg_amount", F.avg("invoice_amount").over(w))
               .drop("days_late", "invoice_amount"))

    # A customer can have several invoices clear on the same day, which would make
    # (customer_id, clear_date) non-unique - rejected as a feature-table primary key,
    # and worse, it made asof_join_history's tie-break arbitrary: it could pick the
    # state after the first of the day's payments and silently undercount history.
    # Keep the last row of each day, i.e. the state once all of it has settled.
    settled = (Window.partitionBy("customer_id", "clear_date")
               .orderBy(F.col("cust_invoice_count").desc()))
    return (running
            .withColumn("_r", F.row_number().over(settled))
            .filter(F.col("_r") == 1)
            .drop("_r"))


def asof_join_history(invoices: DataFrame, timeline: DataFrame, defaults: dict) -> DataFrame:
    """Attach each invoice the customer state as of the moment it was raised.

    Strictly `clear_date < posting_date`: an invoice must not see a payment that
    cleared the same day it was raised. This mirrors merge_asof(allow_exact_matches=False)
    in the pandas pipeline - dropping the strictness is a silent same-day leak.

    Equivalent to a Feature Engineering client point-in-time lookup; kept as plain
    Spark so it runs and can be tested without the Databricks runtime.
    """
    t = timeline.select("customer_id",
                        F.col("clear_date").alias("_hist_asof"),
                        *HIST_COLS)
    joined = invoices.join(
        t,
        (invoices.customer_id == t.customer_id) & (t._hist_asof < invoices.posting_date),
        how="left",
    ).drop(t.customer_id)

    w = Window.partitionBy("invoice_id").orderBy(F.col("_hist_asof").desc_nulls_last())
    latest = (joined.withColumn("_rn", F.row_number().over(w))
              .filter(F.col("_rn") == 1)
              .drop("_rn", "_hist_asof"))

    for col, value in defaults.items():
        latest = latest.withColumn(col, F.coalesce(F.col(col), F.lit(float(value))))
    return latest


def add_invoice_features(df: DataFrame) -> DataFrame:
    """Non-history features. All knowable at invoice creation, so no leakage risk.

    day_of_week is converted to pandas' convention (0=Monday .. 6=Sunday). Spark's
    dayofweek() is 1=Sunday .. 7=Saturday, so leaving it raw would give the same
    invoice a different value here than in deploy/src/features.py - a silent
    train/serve skew the moment a model crosses between the two pipelines.
    """
    return (df
            .withColumn("amount_log", F.log1p(F.abs(F.col("invoice_amount"))))
            .withColumn("month", F.month("posting_date"))
            .withColumn("day_of_week", F.pmod(F.dayofweek("posting_date") + 5, F.lit(7)))
            .withColumn("is_month_end", (F.dayofmonth("posting_date") > 25).cast("int"))
            .withColumn("is_year_end", (F.month("posting_date") == 12).cast("int")))
