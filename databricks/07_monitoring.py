# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Monitoring with delayed labels (Phase 8)
# MAGIC
# MAGIC At prediction time the answer does not exist. It arrives weeks later, when the invoice
# MAGIC is finally paid. That gap is the whole problem this notebook exists to handle.
# MAGIC
# MAGIC ```
# MAGIC posting_date          ...weeks...          clear_date
# MAGIC      |                                          |
# MAGIC   predict  ---------> logged prediction <---- outcome joins back
# MAGIC ```
# MAGIC
# MAGIC ## Read this before trusting any number below
# MAGIC
# MAGIC Performance is measured **only on `backtest` rows** — closed invoices from the model's
# MAGIC held-out window, re-scored using features available at their posting date. Their real
# MAGIC payment outcomes are genuine labels.
# MAGIC
# MAGIC The `open` rows in the prediction log **never acquire labels**: those invoices are
# MAGIC unpaid in the source data and always will be. They are monitored for *drift* only.
# MAGIC Inventing payment dates for them would produce accuracy metrics measuring fabricated
# MAGIC data.
# MAGIC
# MAGIC ## What cannot be measured here
# MAGIC
# MAGIC No collection calls were ever actually made, so **intervention effectiveness is
# MAGIC unmeasurable**. `DAYS_ACCELERATED` and `P(responds)` remain assumptions. Establishing
# MAGIC them needs a holdout where some queued invoices are deliberately not called.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "payment_ops")
dbutils.widgets.text("model_alias", "champion")

# COMMAND ----------

%load_ext autoreload
%autoreload 2

# COMMAND ----------

import os, sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import functions as F, Window

from config import Paths
from modeling import (categorical_psi, drift_label, model_metrics,
                      population_stability_index)

P = Paths(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
ALIAS = dbutils.widgets.get("model_alias")
MODEL_NAME = f"{P.catalog}.{P.schema}.payment_lateness"
PREDICTIONS = f"{P.catalog}.{P.schema}.gold_invoice_predictions"
PERFORMANCE = f"{P.catalog}.{P.schema}.gold_model_performance"
DRIFT = f"{P.catalog}.{P.schema}.gold_drift_metrics"

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. The delayed-label join
# MAGIC
# MAGIC Left join, so predictions still waiting on an outcome survive with a null label. The
# MAGIC count of those is itself a metric — it tells you how much of your recent scoring you
# MAGIC cannot yet evaluate.

# COMMAND ----------

latest = (spark.table(PREDICTIONS)
          .withColumn("_rn", F.row_number().over(
              Window.partitionBy("invoice_id", "scoring_mode").orderBy(F.col("scored_at").desc())))
          .filter(F.col("_rn") == 1).drop("_rn"))

payments = spark.table(P.table("silver_payments")).select("invoice_id", "clear_date")

joined = (latest.join(payments, on="invoice_id", how="left")
          .withColumn("actual_days_late", F.datediff("clear_date", "due_date"))
          .withColumn("has_label", F.col("clear_date").isNotNull())
          .withColumn("label_lag_days", F.datediff("clear_date", "posting_date")))

display(joined.groupBy("scoring_mode", "has_label").count().orderBy("scoring_mode"))

# COMMAND ----------

pdf = joined.toPandas()
for c in ("posting_date", "due_date", "clear_date", "scored_at"):
    if c in pdf.columns:
        pdf[c] = pd.to_datetime(pdf[c])

labelled = pdf[pdf.has_label].copy()
unlabelled = pdf[~pdf.has_label].copy()

print(f"predictions in log     : {len(pdf):,}")
print(f"  with outcome         : {len(labelled):,}")
print(f"  awaiting outcome     : {len(unlabelled):,}")
if len(labelled):
    print(f"\nlabel arrival lag (posting -> clear), days:")
    print(labelled.label_lag_days.describe().round(1).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC The lag distribution is the operational reality of this problem: you wait roughly a
# MAGIC month to learn whether a prediction was any good. Any retraining cadence faster than
# MAGIC that is training on incomplete feedback.

# COMMAND ----------

if len(labelled):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.hist(labelled.label_lag_days.clip(0, 120), bins=60)
    ax.set_xlabel("days from posting to payment"); ax.set_ylabel("invoices")
    ax.set_title("How long until the label arrives")
    plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model performance once outcomes arrived
# MAGIC
# MAGIC Compared against the metrics logged at training time. A meaningful gap is decay.

# COMMAND ----------

backtest = labelled[labelled.scoring_mode == "backtest"]
assert len(backtest) > 0, (
    "no labelled backtest predictions - re-run 05_batch_inference with "
    "score_scope = 'backtest' or 'both'")

live = model_metrics(backtest.actual_days_late, backtest.days_late_pred,
                     backtest.days_late_lower, backtest.days_late_upper)

from mlflow.tracking import MlflowClient
client = MlflowClient()
mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
train_metrics = client.get_run(mv.run_id).data.metrics

rows = []
for k in ["mae", "rmse", "bias", "within_3d", "pi_coverage", "late_auc"]:
    t, l = train_metrics.get(k), live.get(k)
    rows.append({"metric": k, "at_training": t, "on_delayed_labels": l,
                 "delta": (l - t) if (t is not None and l is not None) else None})
comparison = pd.DataFrame(rows)
display(comparison.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC `pi_coverage` is the one to watch. If the interval stops covering ~80% of outcomes, the
# MAGIC decision engine's `P(late)` — which is derived from the interval, not the point
# MAGIC estimate — silently degrades, and the ranking degrades with it. That failure is
# MAGIC invisible if you only track MAE.

# COMMAND ----------

resid = backtest.days_late_pred - backtest.actual_days_late
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
ax[0].scatter(backtest.actual_days_late, backtest.days_late_pred, s=4, alpha=.2)
ax[0].plot([-40, 60], [-40, 60], "k--", lw=1); ax[0].set_xlim(-40, 60); ax[0].set_ylim(-40, 60)
ax[0].set_xlabel("actual"); ax[0].set_ylabel("predicted"); ax[0].set_title("Delayed labels: predicted vs actual")
ax[1].hist(np.clip(resid, -40, 40), bins=70); ax[1].axvline(0, c="k", ls="--", lw=1)
ax[1].set_title(f"Residuals (bias {live['bias']:+.2f})")
inside = ((backtest.actual_days_late >= backtest.days_late_lower)
          & (backtest.actual_days_late <= backtest.days_late_upper))
by_month = backtest.assign(inside=inside, m=backtest.posting_date.dt.to_period("M").astype(str))
cov = by_month.groupby("m").inside.mean() * 100
ax[2].plot(cov.index, cov.values, marker="o"); ax[2].axhline(80, c="r", ls="--", lw=1)
ax[2].set_title("PI coverage by month"); ax[2].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Drift — data and prediction
# MAGIC
# MAGIC Reference is the training population; current is what is being scored now. PSI bins come
# MAGIC from the reference, so the question is "where does today's traffic sit relative to what
# MAGIC the model was fitted on".

# COMMAND ----------

train_ref = spark.table(f"{P.catalog}.{P.schema}.gold_training_dataset").toPandas()
current = pdf[pdf.scoring_mode == "open"] if (pdf.scoring_mode == "open").any() else pdf

numeric = ["invoice_amount", "amount_log", "cust_avg_days_late", "cust_std_days_late",
           "cust_invoice_count"]
drift_rows = []
for c in numeric:
    if c in train_ref.columns and c in current.columns:
        psi = population_stability_index(train_ref[c], current[c])
        drift_rows.append({"feature": c, "type": "numeric", "psi": psi, "status": drift_label(psi)})
for c in ["payment_terms", "business_code"]:
    if c in train_ref.columns and c in current.columns:
        psi = categorical_psi(train_ref[c], current[c])
        drift_rows.append({"feature": c, "type": "categorical", "psi": psi, "status": drift_label(psi)})

drift_df = pd.DataFrame(drift_rows).sort_values("psi", ascending=False)
display(drift_df.round(4))

# COMMAND ----------

# Prediction drift: is the model's OUTPUT distribution moving, regardless of inputs?
pred_psi = population_stability_index(
    train_ref["days_late"] if "days_late" in train_ref.columns else backtest.actual_days_late,
    current.days_late_pred)
print(f"prediction distribution PSI : {pred_psi:.4f}  ({drift_label(pred_psi)})")
print(f"mean predicted days late    : {current.days_late_pred.mean():.2f}")
print(f"P90 predicted days late     : {current.days_late_pred.quantile(0.9):.2f}")
print(f"mean P(late)                : {current.p_late.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Some drift is expected and is not a defect: the open book is later in time than
# MAGIC anything trained on, and spans Feb–May 2020. Attributing that to COVID is plausible but
# MAGIC unproven — report it as observed shift, not as a diagnosed cause.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Business monitoring
# MAGIC
# MAGIC For queued invoices that have since been paid: was the invoice actually late, and did
# MAGIC the tier the engine assigned make sense in hindsight?
# MAGIC
# MAGIC This measures **whether the ranking targeted the right invoices**, not whether calling
# MAGIC them helped. The second question needs an experiment.

# COMMAND ----------

queue_tbl = f"{P.catalog}.{P.schema}.gold_collection_queue"
if spark.catalog.tableExists(queue_tbl) and len(backtest):
    # Re-derive tiers for backtest rows so hindsight can be applied to them.
    hind = backtest.assign(
        predicted_late=lambda d: d.days_late_pred > 0,
        actually_late=lambda d: d.actual_days_late > 0)
    ct = pd.crosstab(hind.predicted_late, hind.actually_late)
    print("predicted late (rows) vs actually late (cols):")
    print(ct.to_string())
    tp = int(((hind.predicted_late) & (hind.actually_late)).sum())
    fp = int(((hind.predicted_late) & (~hind.actually_late)).sum())
    fn = int(((~hind.predicted_late) & (hind.actually_late)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    print(f"\nprecision (of those flagged late, how many were): {prec:.3f}")
    print(f"recall    (of those actually late, how many caught): {rec:.3f}")
    print("\nA false positive costs one $15 call. A false negative costs delayed cash on a")
    print("real late invoice - the asymmetry is why the engine ranks by value, not accuracy.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Persist
# MAGIC
# MAGIC Appended so the trend across runs is visible — a single snapshot cannot show decay.

# COMMAND ----------

perf = pd.DataFrame([{**{k: float(v) for k, v in live.items()},
                      "model_name": MODEL_NAME, "model_version": str(mv.version),
                      "n_labelled": len(backtest), "n_awaiting_label": len(unlabelled),
                      "median_label_lag_days": float(labelled.label_lag_days.median()) if len(labelled) else None,
                      "evaluated_at": pd.Timestamp.utcnow().tz_localize(None)}])
(spark.createDataFrame(perf).write.mode("append").option("mergeSchema", "true").saveAsTable(PERFORMANCE))

drift_out = drift_df.assign(model_version=str(mv.version),
                            evaluated_at=pd.Timestamp.utcnow().tz_localize(None))
(spark.createDataFrame(drift_out).write.mode("append").option("mergeSchema", "true").saveAsTable(DRIFT))

print("wrote", PERFORMANCE, "and", DRIFT)
display(spark.table(PERFORMANCE).orderBy(F.col("evaluated_at").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Phase 8 done
# MAGIC
# MAGIC | table | contents |
# MAGIC |---|---|
# MAGIC | `gold_model_performance` | accuracy on delayed labels, per run |
# MAGIC | `gold_drift_metrics` | PSI per feature, per run |
# MAGIC
# MAGIC **Honest summary for the README:** performance is measured on a backtest against real
# MAGIC outcomes; the open book cannot be scored because those invoices never clear in this
# MAGIC dataset; intervention effectiveness is assumed, never measured.
# MAGIC
# MAGIC **Next (Phase 9):** dashboard over the gold tables.
