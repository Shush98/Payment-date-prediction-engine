# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver: quality, dedup, validation (Phase 3)
# MAGIC
# MAGIC Bronze is faithful but untrustworthy. Silver makes it usable:
# MAGIC
# MAGIC 1. **Deduplicate** on the producer-assigned `_event_id`
# MAGIC 2. **Enforce schema** — parse dates and amounts, don't guess
# MAGIC 3. **Validate** business rules, quarantine failures with a reason
# MAGIC 4. **Join outcomes** — payments onto invoices, producing the `days_late` label
# MAGIC 5. **Record quality metrics** for every run
# MAGIC
# MAGIC Bad rows are **quarantined, not dropped**. A silently discarded row is a bug you find
# MAGIC three months later; a quarantine table with a reason column is one you find today.
# MAGIC
# MAGIC The transformation logic lives in `transforms.py` and is unit-tested locally against
# MAGIC plain pyspark (`python databricks/test_transforms.py`).

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "payment_ops")

# COMMAND ----------

import os, sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pyspark.sql import functions as F
from config import Paths
from transforms import (dedupe_events, validate_invoices, validate_payments,
                        join_outcomes, data_quality_metrics)

P = Paths(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
bronze = spark.table(P.table("bronze_events"))
rows_in = bronze.count()
print(f"bronze rows: {rows_in:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Deduplicate
# MAGIC
# MAGIC Keyed on `_event_id`, keeping the latest ingestion. Deduping on the whole row would let
# MAGIC a replayed event through, because `_ingested_at` differs on the second delivery.

# COMMAND ----------

events = dedupe_events(bronze)
deduped = events.count()
print(f"{rows_in:,} -> {deduped:,}  ({rows_in - deduped:,} duplicate events collapsed)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 & 3. Validate and split
# MAGIC
# MAGIC | rule | reason code |
# MAGIC |---|---|
# MAGIC | missing invoice or customer id | `null_invoice_id`, `null_customer_id` |
# MAGIC | unparseable dates | `unparseable_posting_date`, `unparseable_due_date` |
# MAGIC | due date before posting date | `due_before_posting` |
# MAGIC | missing amount | `null_amount` |
# MAGIC | amount beyond plausible range | `implausible_amount` |
# MAGIC
# MAGIC Negative amounts are **allowed** — they are credit notes, not errors. Discarding them
# MAGIC would quietly bias the amount distribution.

# COMMAND ----------

inv_clean, inv_bad = validate_invoices(events)
pay_clean, pay_bad = validate_payments(events)

print(f"invoices: {inv_clean.count():,} clean / {inv_bad.count():,} quarantined")
print(f"payments: {pay_clean.count():,} clean / {pay_bad.count():,} quarantined")
display(inv_bad.groupBy("_reject_reason").count())

# COMMAND ----------

INVOICE_COLS = ["invoice_id", "customer_id", "customer_name", "invoice_amount",
                "posting_date", "due_date", "payment_terms", "business_code",
                "_event_id", "_source", "_ingested_at"]

silver_invoices = inv_clean.select(*INVOICE_COLS).dropDuplicates(["invoice_id"])
silver_payments = pay_clean.select("invoice_id", "clear_date", "_event_id", "_ingested_at") \
                           .dropDuplicates(["invoice_id"])

(silver_invoices.write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(P.table("silver_invoices")))
(silver_payments.write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(P.table("silver_payments")))

print("silver_invoices:", spark.table(P.table("silver_invoices")).count())
print("silver_payments:", spark.table(P.table("silver_payments")).count())

# COMMAND ----------

# MAGIC %md
# MAGIC `dropDuplicates(["invoice_id"])` above is a second, coarser net. The source data has
# MAGIC ~1,167 repeated `invoice_id` values that are *not* replayed events — they are genuine
# MAGIC duplicates in the original CSV. Deduping on `_event_id` alone would keep them.

# COMMAND ----------

quarantine = (inv_bad.select("_event_id", "event_type", "invoice_id", "_reject_reason", "_ingested_at")
              .unionByName(pay_bad.select("_event_id", "event_type", "invoice_id",
                                          "_reject_reason", "_ingested_at")))
quarantine.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(P.table("quarantine"))
print("quarantined rows:", spark.table(P.table("quarantine")).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Join outcomes — the delayed label
# MAGIC
# MAGIC A left join, so invoices with no payment yet survive with a null `clear_date`. Those are
# MAGIC the open book the model must predict. `days_late = clear_date - due_date`, negative when
# MAGIC paid early.
# MAGIC
# MAGIC This join is the delayed-label mechanism: as more `invoice_paid` events arrive, rows that
# MAGIC were open acquire labels, and predictions made earlier become scoreable.

# COMMAND ----------

outcomes = join_outcomes(silver_invoices, silver_payments)
(outcomes.write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(P.table("silver_outcomes")))

o = spark.table(P.table("silver_outcomes"))
print("total invoices :", o.count())
print("labelled (paid):", o.filter("is_open = 0").count())
print("open           :", o.filter("is_open = 1").count())
display(o.filter("is_open = 0").select(F.avg("days_late").alias("avg_days_late"),
                                       F.expr("percentile_approx(days_late, 0.5)").alias("median"),
                                       F.avg((F.col("days_late") > 0).cast("double")).alias("late_rate")))

# COMMAND ----------

# MAGIC %md
# MAGIC Sanity check against the known figures from the local pipeline: mean `days_late` ≈ 0.84,
# MAGIC median 0, late rate ≈ 41.9%. If the stream has only replayed part of the source these
# MAGIC will differ — that is expected, not a failure.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quality metrics
# MAGIC
# MAGIC Appended, not overwritten, so the trend across runs is visible. This table is what the
# MAGIC Phase 8 monitoring dashboard reads.

# COMMAND ----------

metrics = data_quality_metrics(spark, "silver_invoices", rows_in, inv_clean, inv_bad)
(metrics.write.mode("append").option("mergeSchema", "true").saveAsTable(P.table("dq_metrics")))
display(spark.table(P.table("dq_metrics")).orderBy(F.col("run_timestamp").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guardrail
# MAGIC
# MAGIC Fails the notebook rather than letting a broken Silver flow into features.

# COMMAND ----------

assert o.filter("invoice_id IS NULL").count() == 0, "null invoice_id reached silver"
assert o.groupBy("invoice_id").count().filter("count > 1").count() == 0, "duplicate invoice_id in silver"
assert o.filter("due_date < posting_date").count() == 0, "invalid date ordering reached silver"
print("silver guardrails passed")
