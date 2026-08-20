# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Batch inference on the open book (Phase 6)
# MAGIC
# MAGIC Scores every unpaid invoice with the registered `champion` model and appends to a
# MAGIC **prediction log**.
# MAGIC
# MAGIC ## Why a log rather than a snapshot
# MAGIC
# MAGIC The table is `append`, not `overwrite`, and every row records *when* it was scored and
# MAGIC by *which model version*. That is what makes Phase 8 possible: when an invoice is
# MAGIC eventually paid, its `invoice_paid` event joins back to the prediction that was made
# MAGIC before the outcome existed. Overwriting would destroy the evidence needed to score the
# MAGIC model honestly.
# MAGIC
# MAGIC ## Features at inference time
# MAGIC
# MAGIC The timeline is built from **all** cleared invoices here, not just a train fold. That is
# MAGIC correct: at scoring time every one of those outcomes has genuinely happened. The
# MAGIC train-fold-only rule applies to *training*, where future outcomes must be withheld.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "payment_ops")
dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.dropdown("score_scope", "both", ["open", "backtest", "both"])

# COMMAND ----------

%load_ext autoreload
%autoreload 2

# COMMAND ----------

import os, sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from config import Paths
from modeling import DEFAULT_KEYS, RAW_INPUT_COLS
from transforms import add_invoice_features, asof_join_history, build_customer_timeline

P = Paths(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
ALIAS = dbutils.widgets.get("model_alias")
MODEL_NAME = f"{P.catalog}.{P.schema}.payment_lateness"
PREDICTIONS = f"{P.catalog}.{P.schema}.gold_invoice_predictions"

mlflow.set_registry_uri("databricks-uc")
print(f"model: {MODEL_NAME}@{ALIAS}\npredictions -> {PREDICTIONS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the model and recover its training-time defaults
# MAGIC
# MAGIC The cold-start fallbacks were logged as `default_*` params on the training run. Reading
# MAGIC them from the run that produced *this* model version keeps features consistent with
# MAGIC training — recomputing them here would drift as new data arrives.

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
run = client.get_run(mv.run_id)

defaults = {k: float(run.data.params[f"default_{k}"]) for k in DEFAULT_KEYS}
MODEL_VERSION = mv.version

print(f"version {MODEL_VERSION}  (run {mv.run_id})")
print("training-time defaults:")
for k, v in defaults.items():
    print(f"  {k:<22}{v:>12.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build features for the open book

# COMMAND ----------

outcomes = spark.table(P.table("silver_outcomes"))
timeline = build_customer_timeline(outcomes)          # all cleared invoices - correct at scoring time

# --- what to score -----------------------------------------------------------------
# `open`     the real open book. These invoices never clear in this dataset, so they
#            can never produce a label - useful for the queue, useless for monitoring.
# `backtest` invoices from the model's held-out test window. They are closed, so their
#            real payment outcome supplies a GENUINE delayed label for Phase 8. Features
#            still come from the as-of join, so nothing later than posting_date is used.
# Fabricating payments for the open book would fill the dashboard with numbers measuring
# invented data - see the plan's "do not present simulated results as observed" rule.
SCOPE = dbutils.widgets.get("score_scope")
cutoff = pd.Timestamp(run.data.params["cutoff_date"]).date()

frames = []
if SCOPE in ("open", "both"):
    frames.append(("open", outcomes.filter("is_open = 1")))
if SCOPE in ("backtest", "both"):
    frames.append(("backtest", outcomes.filter(
        (F.col("is_open") == 0) & (F.col("posting_date") >= F.lit(cutoff)))))

parts = []
for mode, sdf in frames:
    feat = add_invoice_features(asof_join_history(sdf, timeline, defaults))
    pdf_part = feat.select("invoice_id", "customer_id", "customer_name", "invoice_amount",
                           "posting_date", "due_date", *RAW_INPUT_COLS).toPandas()
    pdf_part["scoring_mode"] = mode
    print(f"  {mode:<9} {len(pdf_part):>7,} invoices")
    parts.append(pdf_part)

score_pdf = pd.concat(parts, ignore_index=True)
for col in ("posting_date", "due_date"):
    score_pdf[col] = pd.to_datetime(score_pdf[col])

print(f"\ntotal to score: {len(score_pdf):,}   (model test window starts {cutoff})")
print("cold-start (no prior cleared history):", int((score_pdf.cust_invoice_count == 0).sum()))

# COMMAND ----------

assert len(score_pdf) > 0, "nothing to score - run simulation/invoice_stream.py for more batches"
missing = [c for c in RAW_INPUT_COLS if c not in score_pdf.columns]
assert not missing, f"missing model inputs: {missing}"
if SCOPE in ("open", "both"):
    assert (score_pdf.scoring_mode == "open").sum() > 0, (
        "no OPEN invoices - the generator emits in posting-date order and the open book is "
        "the most recent ~10k rows; raise batches x batch_size and re-run 01-03")

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
preds = model.predict(score_pdf[RAW_INPUT_COLS])
print(preds.describe().round(2).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assemble the prediction log
# MAGIC
# MAGIC `predicted_payment_date` uses the ceiling of the point estimate: a collections team
# MAGIC needs a whole day, and rounding down would systematically promise money early.

# COMMAND ----------

log = score_pdf[["invoice_id", "customer_id", "customer_name", "invoice_amount",
                 "posting_date", "due_date", "cust_std_days_late", "cust_invoice_count",
                 "amount_log", "payment_terms", "business_code", "scoring_mode"]].copy()
log["days_late_pred"] = preds["days_late_pred"].to_numpy()
log["days_late_lower"] = preds["days_late_lower"].to_numpy()
log["days_late_upper"] = preds["days_late_upper"].to_numpy()
log["p_late"] = preds["p_late"].to_numpy()
log["predicted_payment_date"] = (
    log["due_date"] + pd.to_timedelta(np.ceil(log["days_late_pred"]).astype(int), unit="D"))

log["model_name"] = MODEL_NAME
log["model_version"] = str(MODEL_VERSION)
log["model_run_id"] = mv.run_id
log["scored_at"] = pd.Timestamp.utcnow().tz_localize(None)

assert (log.days_late_lower <= log.days_late_upper).all(), "inverted prediction interval"
display(log.head(10))

# COMMAND ----------

(spark.createDataFrame(log)
 .write.mode("append").option("mergeSchema", "true").saveAsTable(PREDICTIONS))

total = spark.table(PREDICTIONS)
print(f"appended {len(log):,} rows  |  prediction log now {total.count():,} rows")
display(total.groupBy("model_version", "scored_at").count().orderBy(F.col("scored_at").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribution check
# MAGIC
# MAGIC The open book is later in time than everything trained on, so some shift is expected
# MAGIC rather than alarming. A **large** gap between predicted late rate and the historical
# MAGIC base rate is the drift signal Phase 8 will monitor properly.

# COMMAND ----------

hist_late_rate = (outcomes.filter("is_open = 0")
                  .select(F.avg((F.col("days_late") > 0).cast("double"))).first()[0])
pred_late_rate = float((log.days_late_pred > 0).mean())

print(f"historical late rate (closed) : {hist_late_rate*100:>5.1f}%")
print(f"predicted late rate (open)    : {pred_late_rate*100:>5.1f}%")
print(f"mean predicted days late      : {log.days_late_pred.mean():>5.2f}")
print(f"mean interval width           : {(log.days_late_upper - log.days_late_lower).mean():>5.2f} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Phase 6 done — next notebook turns these predictions into decisions.
