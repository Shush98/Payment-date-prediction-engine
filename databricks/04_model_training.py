# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Training + MLflow + Unity Catalog registry (Phase 5)
# MAGIC
# MAGIC Trains the three quantile models, logs everything to MLflow, and registers a single
# MAGIC pyfunc model in Unity Catalog.
# MAGIC
# MAGIC ## The one thing this notebook must not get wrong
# MAGIC
# MAGIC `gold_training_dataset` from notebook 03 was built with a timeline over **all** of
# MAGIC Silver. That is correct for *scoring* today's open invoices — every outcome in it has
# MAGIC genuinely happened. It is **wrong for training**, because the test fold's outcomes
# MAGIC would sit inside the training rows' features.
# MAGIC
# MAGIC So this notebook goes back to `silver_invoice_outcomes`, splits chronologically, and
# MAGIC **rebuilds the timeline from the train fold only**. Reading `gold_training_dataset` and
# MAGIC splitting it would silently reintroduce the leak the whole project exists to fix.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "payment_ops")
dbutils.widgets.text("test_fraction", "0.2")
dbutils.widgets.dropdown("register_model", "true", ["true", "false"])

# COMMAND ----------

# Picks up edits to transforms.py / modeling.py after a git pull.
%load_ext autoreload
%autoreload 2

# COMMAND ----------

import os, sys
from datetime import date, timedelta

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.ensemble import GradientBoostingRegressor

from config import Paths
from modeling import (FEATURE_COLS, PaymentLatenessModel, apply_encoders, build_encoders,
                      business_metrics, model_metrics)
from transforms import HIST_COLS, add_invoice_features, asof_join_history, build_customer_timeline

P = Paths(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
TEST_FRACTION = float(dbutils.widgets.get("test_fraction"))
REGISTER = dbutils.widgets.get("register_model") == "true"
MODEL_NAME = f"{P.catalog}.{P.schema}.payment_lateness"
GBM = dict(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
print("model:", MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chronological split
# MAGIC
# MAGIC The cutoff is found with `approxQuantile` on the posting date rather than a
# MAGIC `row_number()` window — an unpartitioned window would funnel every row through one
# MAGIC partition for no benefit.

# COMMAND ----------

EPOCH = date(1970, 1, 1)
labelled = spark.table(P.table("silver_outcomes")).filter("is_open = 0 AND days_late IS NOT NULL")

as_days = labelled.select(F.datediff("posting_date", F.lit(EPOCH)).alias("d"))
cut = as_days.approxQuantile("d", [1 - TEST_FRACTION], 0.0001)[0]
CUTOFF = EPOCH + timedelta(days=int(cut))

train_sdf = labelled.filter(F.col("posting_date") < F.lit(CUTOFF))
test_sdf = labelled.filter(F.col("posting_date") >= F.lit(CUTOFF))

print(f"cutoff: {CUTOFF}")
for name, d in [("train", train_sdf), ("test ", test_sdf)]:
    r = d.select(F.min("posting_date").alias("lo"), F.max("posting_date").alias("hi")).first()
    print(f"  {name}: {d.count():>6,}   {r['lo']} .. {r['hi']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features — timeline from the train fold only

# COMMAND ----------

d = train_sdf.agg(
    F.avg("days_late").alias("avg"), F.stddev("days_late").alias("std"),
    F.min("days_late").alias("min"), F.max("days_late").alias("max"),
    F.avg("invoice_amount").alias("amt"),
).first()

defaults = {
    "cust_avg_days_late": float(d["avg"] or 0.0),
    "cust_std_days_late": float(d["std"] or 0.0),
    "cust_invoice_count": 0.0,
    "cust_min_late": float(d["min"] or 0.0),
    "cust_max_late": float(d["max"] or 0.0),
    "cust_avg_amount": float(d["amt"] or 0.0),
}

train_timeline = build_customer_timeline(train_sdf)      # TRAIN FOLD ONLY - see header
f_train = add_invoice_features(asof_join_history(train_sdf, train_timeline, defaults))
f_test = add_invoice_features(asof_join_history(test_sdf, train_timeline, defaults))

KEEP = FEATURE_COLS + ["payment_terms", "business_code", "invoice_amount",
                       "cust_std_days_late", "posting_date", "days_late", "invoice_id"]
train_pdf = f_train.select(*[c for c in dict.fromkeys(KEEP) if c in f_train.columns]).toPandas()
test_pdf = f_test.select(*[c for c in dict.fromkeys(KEEP) if c in f_test.columns]).toPandas()
print(f"train {train_pdf.shape}   test {test_pdf.shape}")

# COMMAND ----------

# Leakage guard: a test outcome must never have reached a training feature. The train
# timeline cannot contain any clear_date from the test fold by construction, but assert
# it rather than trust it - this is the failure the whole project is about.
max_train_clear = train_sdf.agg(F.max("clear_date")).first()[0]
leaked = train_timeline.filter(F.col("clear_date") > F.lit(max_train_clear)).count()
assert leaked == 0, f"{leaked} timeline rows come from beyond the training window"
assert train_pdf.posting_date.max() < test_pdf.posting_date.min(), "folds overlap in time"
print("leakage guards passed")

# COMMAND ----------

encoders = build_encoders(train_pdf)
train_pdf = apply_encoders(train_pdf, encoders)
test_pdf = apply_encoders(test_pdf, encoders)

X_train, y_train = train_pdf[FEATURE_COLS].to_numpy(), train_pdf["days_late"].to_numpy()
X_test, y_test = test_pdf[FEATURE_COLS].to_numpy(), test_pdf["days_late"].to_numpy()
print("unseen payment terms in test:", int((test_pdf.payment_terms_encoded == -1).sum()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and log
# MAGIC
# MAGIC One MLflow run holds all three models plus the metrics. Business metrics are logged
# MAGIC alongside the statistical ones because accuracy is not the objective here — expected
# MAGIC value under a capacity constraint is.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name=f"gbm-quantile-{CUTOFF}") as run:
    model_mean = GradientBoostingRegressor(**GBM).fit(X_train, y_train)
    model_lower = GradientBoostingRegressor(loss="quantile", alpha=0.1, **GBM).fit(X_train, y_train)
    model_upper = GradientBoostingRegressor(loss="quantile", alpha=0.9, **GBM).fit(X_train, y_train)

    bundle = PaymentLatenessModel(model_mean, model_lower, model_upper)
    preds = bundle.predict(test_pdf)

    stats = model_metrics(y_test, preds.days_late_pred, preds.days_late_lower, preds.days_late_upper)
    biz = business_metrics(test_pdf, preds.days_late_lower, preds.days_late_upper)

    mlflow.log_params({
        **{f"gbm_{k}": v for k, v in GBM.items()},
        "lower_quantile": 0.1, "upper_quantile": 0.9,
        "split": "chronological", "test_fraction": TEST_FRACTION, "cutoff_date": str(CUTOFF),
        "n_train": len(X_train), "n_test": len(X_test),
        "n_features": len(FEATURE_COLS),
        "feature_source": "point_in_time_timeline_train_fold_only",
    })
    mlflow.log_metrics({**stats, **biz})
    mlflow.set_tags({
        "silver_table": P.table("silver_outcomes"),
        "feature_table": P.table("feature_timeline"),
        "train_window": f"{train_pdf.posting_date.min().date()}..{train_pdf.posting_date.max().date()}",
        "test_window": f"{test_pdf.posting_date.min().date()}..{test_pdf.posting_date.max().date()}",
        "leakage_controls": "point-in-time as-of join; timeline built from train fold only",
    })

    # --- plots ---
    resid = preds.days_late_pred.to_numpy() - y_test
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].scatter(y_test, preds.days_late_pred, s=3, alpha=.15)
    ax[0].plot([-40, 60], [-40, 60], "k--", lw=1); ax[0].set_xlim(-40, 60); ax[0].set_ylim(-40, 60)
    ax[0].set_xlabel("actual"); ax[0].set_ylabel("predicted"); ax[0].set_title("Predicted vs actual")
    ax[1].hist(np.clip(resid, -40, 40), bins=80); ax[1].axvline(0, c="k", ls="--", lw=1)
    ax[1].set_title(f"Residuals (bias {stats['bias']:+.2f})")
    imp = pd.Series(model_mean.feature_importances_, index=FEATURE_COLS).sort_values()
    ax[2].barh(imp.index, imp.values * 100,
               color=["tab:red" if f in HIST_COLS else "tab:blue" for f in imp.index])
    ax[2].set_title("Importance % (red = customer history)")
    plt.tight_layout()
    mlflow.log_figure(fig, "diagnostics.png")
    plt.close(fig)

    # --- register one pyfunc holding all three models ---
    class PaymentLatenessPyfunc(mlflow.pyfunc.PythonModel):
        def __init__(self, bundle):
            self.bundle = bundle

        def predict(self, context, model_input):
            return self.bundle.predict(model_input)

    signature = mlflow.models.infer_signature(test_pdf[FEATURE_COLS], preds)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=PaymentLatenessPyfunc(bundle),
        signature=signature,
        input_example=test_pdf[FEATURE_COLS].head(3),
        registered_model_name=MODEL_NAME if REGISTER else None,
        pip_requirements=["scikit-learn", "pandas", "numpy"],
    )
    RUN_ID = run.info.run_id

print("run:", RUN_ID)
for k, v in {**stats, **biz}.items():
    print(f"  {k:<24}{v:>12.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the numbers honestly
# MAGIC
# MAGIC | metric | what good looks like |
# MAGIC |---|---|
# MAGIC | `mae` | ~3.2 days. The pre-fix pipeline claimed 2.69 — that number was leakage, not skill |
# MAGIC | `pi_coverage` | near **80**. Below ~76 means overconfident intervals, and the engine's `P(late)` comes from the interval |
# MAGIC | `bias` | near 0 |
# MAGIC | `late_auc` | > 0.5; ranking quality for "will this be late" |
# MAGIC | `ev_decision_engine` | should beat `ev_by_amount` and `ev_random` |
# MAGIC
# MAGIC `ev_*` compares **orderings under the engine's own assumptions**. It is not a measured
# MAGIC business outcome — validating that needs a holdout where some invoices are deliberately
# MAGIC not called. Say so wherever these numbers appear.

# COMMAND ----------

comparison = pd.DataFrame([
    {"pipeline": "pre-fix (leaky features + random split)", "mae": 2.69, "pi_coverage": 76.2, "source": "original artifacts"},
    {"pipeline": "local point-in-time (pandas)", "mae": 3.16, "pi_coverage": 79.3, "source": "deploy/src/train.py"},
    {"pipeline": "databricks point-in-time (this run)", "mae": stats["mae"], "pi_coverage": stats["pi_coverage"], "source": MODEL_NAME},
])
display(comparison.round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC The Databricks and local numbers should be close but need not match exactly — the
# MAGIC stream may have replayed only part of the source, so the folds differ. A large gap
# MAGIC means the two feature pipelines have drifted and is worth chasing.

# COMMAND ----------

if REGISTER:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
    client.update_model_version(
        MODEL_NAME, latest.version,
        description=(f"GBM quantile trio (P10/P50/P90). Chronological split at {CUTOFF}. "
                     f"MAE {stats['mae']:.2f}, PI coverage {stats['pi_coverage']:.1f}%. "
                     "Point-in-time features; timeline built from train fold only."),
    )
    print(f"registered {MODEL_NAME} v{latest.version}, alias 'champion'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load it back
# MAGIC
# MAGIC Proves the registered artifact is usable by something that never saw this notebook —
# MAGIC which is what Phase 6 inference will be.

# COMMAND ----------

if REGISTER:
    loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    sample = test_pdf[FEATURE_COLS].head(5)
    out = loaded.predict(sample)
    print(out)
    assert (out["days_late_lower"] <= out["days_late_upper"]).all(), "interval inverted after round-trip"
    print("\nround-trip OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Phase 5 done
# MAGIC
# MAGIC - Chronological split, timeline rebuilt from the train fold, leakage asserted
# MAGIC - Three quantile models in one registered pyfunc
# MAGIC - Statistical **and** business metrics in MLflow, with diagnostics plot
# MAGIC - Registered in Unity Catalog with a `champion` alias
# MAGIC
# MAGIC **Next (Phase 6):** batch inference over the open book, writing a scored predictions
# MAGIC table, then the decision engine port to produce the ranked collection queue.
