# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Decision engine: the collection queue (Phase 7)
# MAGIC
# MAGIC Turns predictions into decisions. This is the project's actual output — not a number,
# MAGIC a **work order**.
# MAGIC
# MAGIC ## It imports the Flask app's decision engine directly
# MAGIC
# MAGIC `deploy/src/decision.py` is reused as-is rather than reimplemented in Spark, for two
# MAGIC reasons:
# MAGIC
# MAGIC 1. It ranks a few hundred rows. Spark buys nothing and costs serverless quota.
# MAGIC 2. A second implementation would drift. The dashboard and the lakehouse queue must
# MAGIC    rank identically or the demo contradicts itself.
# MAGIC
# MAGIC The engine expects the Flask app's column names, so the only work here is renaming
# MAGIC `invoice_amount` → `total_open_amount` at the boundary.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "payment_ops")
dbutils.widgets.text("daily_capacity", "20")

# COMMAND ----------

%load_ext autoreload
%autoreload 2

# COMMAND ----------

import os, sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
# deploy/ sits alongside databricks/ in the repo; its src package holds the engine.
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "deploy")))

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from config import Paths
from src import decision
from src.decision import build_priority_table, _p_late, _p_responds

P = Paths(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
decision.DAILY_CAPACITY = int(dbutils.widgets.get("daily_capacity"))

PREDICTIONS = f"{P.catalog}.{P.schema}.gold_invoice_predictions"
QUEUE = f"{P.catalog}.{P.schema}.gold_collection_queue"

print("engine parameters")
for k in ["CALL_COST", "DAYS_ACCELERATED", "DAILY_CAPITAL_RATE", "DAILY_CAPACITY",
          "CALL_DELAY_DAYS", "CALL_AMOUNT_USD", "REMIND_DELAY_DAYS"]:
    print(f"  {k:<20} = {getattr(decision, k)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Latest prediction per invoice
# MAGIC
# MAGIC The prediction log is append-only, so an invoice re-scored by a newer model version has
# MAGIC several rows. Today's queue must use the newest.

# COMMAND ----------

from pyspark.sql import Window

# Only genuinely open invoices belong in a work order. The log also holds `backtest`
# rows - closed invoices re-scored to give Phase 8 real labels - and queueing those
# would tell the team to chase invoices that were already paid.
latest = (spark.table(PREDICTIONS)
          .filter(F.col("scoring_mode") == "open")
          .withColumn("_rn", F.row_number().over(
              Window.partitionBy("invoice_id").orderBy(F.col("scored_at").desc())))
          .filter(F.col("_rn") == 1).drop("_rn"))

pdf = latest.toPandas()
print(f"{len(pdf):,} invoices, scored by model version(s): {sorted(pdf.model_version.unique())}")

# COMMAND ----------

# The engine was written against the Flask app's schema. Rename at the boundary rather
# than forking the engine.
pdf = pdf.rename(columns={"invoice_amount": "total_open_amount"})
priority = build_priority_table(pdf)

print(priority.action.value_counts().to_string())
print(f"\nCALL capped at capacity {decision.DAILY_CAPACITY}:",
      int((priority.action == "CALL").sum()) <= decision.DAILY_CAPACITY)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The capacity constraint
# MAGIC
# MAGIC Far more invoices qualify as urgent than the team can call. The engine sorts qualifying
# MAGIC invoices by expected value, keeps the top N as `CALL`, and demotes the rest to `REMIND`
# MAGIC — a cheaper channel rather than nothing.
# MAGIC
# MAGIC Prediction accuracy is not the objective. Expected value delivered under a hard
# MAGIC constraint is.

# COMMAND ----------

priority["p_responds"] = [round(_p_responds(s), 3) for s in priority.cust_std_days_late]
calls = priority[priority.action == "CALL"]

print(f"CALL queue      : {len(calls)} invoices")
print(f"total EV        : ${calls.expected_value.sum():,.2f}")
print(f"value at risk   : ${calls.total_open_amount.sum():,.2f}")
print(f"mean P(late)    : {calls.p_late.mean():.2f}")

display(calls[["rank", "customer_name", "total_open_amount", "days_late_pred",
               "days_late_lower", "days_late_upper", "p_late", "p_responds",
               "expected_value", "action"]].round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Does the ranking beat the obvious alternatives?

# COMMAND ----------

cap = decision.DAILY_CAPACITY
rng = np.random.default_rng(42)
strategies = {
    "Random": float(priority.iloc[rng.choice(len(priority), min(cap, len(priority)),
                                             replace=False)].expected_value.sum()),
    "By amount": float(priority.nlargest(cap, "total_open_amount").expected_value.sum()),
    "Decision engine": float(priority.nlargest(cap, "expected_value").expected_value.sum()),
}
for k, v in strategies.items():
    print(f"  {k:<18} ${v:>12,.2f}")

uplift = strategies["Decision engine"] - strategies["By amount"]
print(f"\nuplift vs by-amount: ${uplift:,.2f}/day  (${uplift*250:,.0f} over 250 working days)")

# COMMAND ----------

# MAGIC %md
# MAGIC **State this carefully.** The comparison is self-consistent under the engine's own
# MAGIC assumptions — the same `P(responds)`, `DAYS_ACCELERATED` and `DAILY_CAPITAL_RATE` are
# MAGIC used to rank and to score. It compares **orderings**, and is not a measured business
# MAGIC outcome. Validating it for real needs a holdout where some invoices are deliberately
# MAGIC not called.

# COMMAND ----------

out = priority.rename(columns={"total_open_amount": "invoice_amount"}).copy()
out["queue_date"] = pd.Timestamp.utcnow().tz_localize(None)

(spark.createDataFrame(out)
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(QUEUE))

print("wrote", QUEUE)
display(spark.table(QUEUE).groupBy("action").agg(
    F.count("*").alias("invoices"),
    F.round(F.sum("expected_value"), 2).alias("total_ev"),
    F.round(F.sum("invoice_amount"), 2).alias("exposure")).orderBy("action"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guardrails

# COMMAND ----------

q = spark.table(QUEUE)
assert q.count() == len(pdf), "queue lost or duplicated invoices"
assert q.filter("action = 'CALL'").count() <= decision.DAILY_CAPACITY, "capacity constraint violated"
assert q.filter("action NOT IN ('CALL','REMIND','WATCH','OK')").count() == 0, "unknown action tier"
assert q.filter("rank IS NULL").count() == 0, "unranked rows in queue"
# Within CALL, rank must follow expected value descending.
call_evs = [r["expected_value"] for r in
            q.filter("action = 'CALL'").orderBy("rank").select("expected_value").collect()]
assert call_evs == sorted(call_evs, reverse=True), "CALL queue not ordered by expected value"
print("queue guardrails passed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Phases 6 + 7 done
# MAGIC
# MAGIC | table | contents |
# MAGIC |---|---|
# MAGIC | `gold_invoice_predictions` | append-only prediction log, stamped with model version |
# MAGIC | `gold_collection_queue` | ranked actions — today's work order |
# MAGIC
# MAGIC **Next (Phase 8):** join `invoice_paid` events back to the prediction log to score the
# MAGIC model on delayed labels, and track drift between predicted and actual.
