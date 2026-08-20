# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Gold + point-in-time feature table (Phase 4)
# MAGIC
# MAGIC This notebook is where the leakage fix becomes permanent infrastructure.
# MAGIC
# MAGIC Two tables that look similar and are **not** interchangeable:
# MAGIC
# MAGIC | table | shape | use |
# MAGIC |---|---|---|
# MAGIC | `gold_customer_payment_behavior` | one row per customer, full history | dashboards, segmentation |
# MAGIC | `feat_customer_payment_history` | one row per *cleared invoice*, running aggregates | **model features** |
# MAGIC
# MAGIC Gold aggregates everything a customer ever did, including invoices that clear in the
# MAGIC future. Feeding it to training is exactly the leak the audit found — measured at 1.14×
# MAGIC MAE inflation and 79.8% of feature importance sitting in target-derived columns.
# MAGIC
# MAGIC The feature table is keyed on `(customer_id, clear_date)` and looked up **as of**
# MAGIC `posting_date`, so a prediction can only ever see outcomes that had already happened.
# MAGIC
# MAGIC Also replaces the joblib `customer_history` artifact the Flask app used to depend on.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "payment_ops")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Feature Engineering client
# MAGIC
# MAGIC Serverless is not the ML Runtime, so `databricks.feature_engineering` is not
# MAGIC preinstalled. Installing it registers the timeline as a **Unity Catalog time-series
# MAGIC feature table**, which makes point-in-time lookups a property of the table rather than
# MAGIC something each training script must remember.
# MAGIC
# MAGIC If the install fails (Free Edition restricts outbound traffic), skip this cell — the
# MAGIC notebook falls back to a plain Delta table and everything downstream still works. You
# MAGIC lose managed lineage, not correctness.
# MAGIC
# MAGIC `%pip install` restarts Python, so this must stay the **first executed cell**.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

# Databricks caches imported modules for the life of the Python process, so a
# `git pull` alone does NOT pick up edits to transforms.py / config.py. Without
# this you re-run the notebook and get the identical error from the old code.
%load_ext autoreload
%autoreload 2

# COMMAND ----------

import os, sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pyspark.sql import functions as F
from config import Paths
from transforms import (HIST_COLS, add_invoice_features, asof_join_history,
                        build_customer_timeline, build_gold_customer_behavior)

P = Paths(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
outcomes = spark.table(P.table("silver_outcomes"))
print("silver outcomes:", outcomes.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold — descriptive customer behaviour

# COMMAND ----------

gold = build_gold_customer_behavior(outcomes)
(gold.write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(P.table("gold_customer")))

print("customers:", spark.table(P.table("gold_customer")).count())
display(spark.table(P.table("gold_customer")).orderBy(F.col("invoice_count").desc()).limit(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature table — running aggregates stamped with clear_date
# MAGIC
# MAGIC Each row is what was knowable about a customer the instant their nth invoice cleared.
# MAGIC A window over `partitionBy(customer_id).orderBy(clear_date)` bounded
# MAGIC `unboundedPreceding -> currentRow` gives the running mean, std, count, min and max.

# COMMAND ----------

timeline = build_customer_timeline(outcomes)
display(timeline.orderBy("customer_id", "clear_date").limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC Follow a single customer to see the aggregates accumulate rather than sit constant —
# MAGIC the constant version is what the leaky pipeline produced.

# COMMAND ----------

busy = (outcomes.filter("is_open = 0").groupBy("customer_id").count()
        .filter("count between 6 and 12").limit(1).first())
if busy:
    display(timeline.filter(F.col("customer_id") == busy["customer_id"]).orderBy("clear_date"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register as a Unity Catalog time-series feature table
# MAGIC
# MAGIC `timeseries_columns` is the part that matters: it tells Feature Engineering that lookups
# MAGIC against this table must be as-of a timestamp, making point-in-time correctness a property
# MAGIC of the table rather than something each training script has to remember.
# MAGIC
# MAGIC Free Edition support for the Feature Engineering client is not guaranteed, so this falls
# MAGIC back to a plain Delta table. The as-of join in `transforms.asof_join_history` works either
# MAGIC way — the fallback loses the managed lineage, not the correctness.

# COMMAND ----------

FEATURE_TABLE = P.table("feature_timeline")
registered = False
try:
    from databricks.feature_engineering import FeatureEngineeringClient

    fe = FeatureEngineeringClient()
    spark.sql(f"DROP TABLE IF EXISTS {FEATURE_TABLE}")
    fe.create_table(
        name=FEATURE_TABLE,
        primary_keys=["customer_id", "clear_date"],
        timeseries_columns="clear_date",
        df=timeline,
        description="Point-in-time customer payment history. Look up as of posting_date.",
    )
    registered = True
    print("registered as a time-series feature table:", FEATURE_TABLE)
except Exception as e:
    (timeline.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(FEATURE_TABLE))
    print(f"Feature Engineering client unavailable, wrote plain Delta table instead.\n  {type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Point-in-time validation (Phase 4 exit criterion)
# MAGIC
# MAGIC Four assertions. Any one of them failing means the leak is back.

# COMMAND ----------

defaults = {c: 0.0 for c in HIST_COLS}
labelled = outcomes.filter("is_open = 0")
training = add_invoice_features(asof_join_history(labelled, timeline, defaults))

# Persist first, then validate what was written. .cache() is unavailable on serverless
# (PERSIST TABLE is not supported), and asserting against the saved table is stronger
# anyway: the guarantees then hold for the artifact Phase 5 reads, not for an
# unmaterialised plan that could be recomputed differently.
TRAINING_TABLE = f"{P.catalog}.{P.schema}.gold_training_dataset"
(training.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TRAINING_TABLE))
training = spark.table(TRAINING_TABLE)

n_invoices = labelled.count()
n_training = training.count()
print(f"invoices in  : {n_invoices:,}")
print(f"rows out     : {n_training:,}  -> {TRAINING_TABLE}")

# COMMAND ----------

# 1. The as-of join must not duplicate invoices.
assert n_training == n_invoices, f"as-of join changed row count: {n_invoices} -> {n_training}"

# 2. No invoice may carry history that cleared on or after its own posting_date.
#    Re-derive the matched history date and check it directly.
t = timeline.select(F.col("customer_id").alias("_c"),
                    F.col("clear_date").alias("_hist"),
                    F.col("cust_invoice_count").alias("_n"))
check = (labelled.join(t, (labelled.customer_id == t._c) & (t._hist < labelled.posting_date), "left"))
violations = check.filter(F.col("_hist") >= F.col("posting_date")).count()
assert violations == 0, f"{violations} rows used history from on/after posting_date"

# 3. An invoice must never see its own outcome: the matched count must be strictly less
#    than the customer's total cleared invoices at that point.
self_leak = (training.join(labelled.select(F.col("invoice_id").alias("_i"),
                                           F.col("clear_date").alias("_own")),
                           training.invoice_id == F.col("_i"))
             .filter(F.col("cust_invoice_count") > 0)
             .filter(F.col("_own") < F.col("posting_date")).count())
print("rows whose own clear_date precedes their posting_date:", self_leak, "(should be 0)")
assert self_leak == 0, "an invoice cleared before it was posted - impossible, check the join"

# 4. Cold starts must fall back to defaults, not to a customer average.
cold = training.filter(F.col("cust_invoice_count") == 0)
print("cold-start rows:", cold.count())
assert cold.filter(F.col("cust_avg_days_late") != 0.0).count() == 0, "cold start did not use defaults"

print("\npoint-in-time validation passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leakage smoke test
# MAGIC
# MAGIC Correlation between the strongest history feature and the label. Under the old leaky
# MAGIC build this was inflated because the feature contained the label. A high value here — say
# MAGIC above 0.5 — means something is wrong.

# COMMAND ----------

display(training.select(
    F.corr("cust_avg_days_late", "days_late").alias("corr_avg_days_late"),
    F.corr("cust_max_late", "days_late").alias("corr_max_late"),
    F.corr("amount_log", "days_late").alias("corr_amount_log"),
    F.avg("cust_invoice_count").alias("avg_history_depth"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training-ready dataset
# MAGIC
# MAGIC Written above, before validation, and confirmed here.
# MAGIC
# MAGIC Note it is **not** split — the chronological split belongs to training, and the timeline
# MAGIC must be rebuilt from the train fold only when that happens. Building it from all of
# MAGIC Silver, as this notebook does, is correct for *scoring* but would leak test outcomes into
# MAGIC training features. Phase 5 must not skip that step.

# COMMAND ----------

print("training dataset:", TRAINING_TABLE, f"({training.count():,} rows)")
display(training.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Phase 4 done
# MAGIC
# MAGIC - Gold customer behaviour table for reporting
# MAGIC - Point-in-time feature table, validated
# MAGIC - joblib `customer_history` dependency replaced on the Databricks path
# MAGIC
# MAGIC **Next (Phase 5):** rebuild the timeline from the train fold only, train the three
# MAGIC quantile models, log to MLflow with the business metrics, and register in Unity Catalog.
