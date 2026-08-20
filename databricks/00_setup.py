# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Lakehouse setup (Phase 1)
# MAGIC
# MAGIC Creates the schema and volumes everything else depends on. Run once.
# MAGIC
# MAGIC **Free Edition notes**
# MAGIC - Serverless only. Exceeding the daily compute quota shuts the workspace down for the rest of the day, so nothing here runs continuously.
# MAGIC - A Unity Catalog metastore exists by default with a catalog named `workspace`.
# MAGIC - Creating a *new catalog* may not be permitted; this notebook only creates a **schema** inside an existing catalog, which is allowed.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "payment_ops")

# COMMAND ----------

# Picks up edits to config.py after a git pull.
%load_ext autoreload
%autoreload 2

# COMMAND ----------

import sys, os

# In a Databricks Git folder the notebook's own directory is already importable.
# This is a no-op fallback for other contexts (e.g. notebook imported standalone).
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from config import Paths, TABLES, VOLUMES

P = Paths(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
print("target:", P)
print("current catalog:", spark.sql("SELECT current_catalog()").first()[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Available catalogs
# MAGIC If `workspace` is not listed below, set the `catalog` widget to one that is.

# COMMAND ----------

display(spark.sql("SHOW CATALOGS"))

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {P.catalog}.{P.schema}")
for v in VOLUMES.values():
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {P.catalog}.{P.schema}.{v}")

print(f"schema ready: {P.catalog}.{P.schema}")
for key in VOLUMES:
    print(f"  volume {key:<12} {P.volume(key)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upload the source data
# MAGIC
# MAGIC The event generator replays real invoices rather than inventing them, so it needs the
# MAGIC original CSV once.
# MAGIC
# MAGIC **Catalog → `workspace` → `payment_ops` → Volumes → `raw` → Upload to this volume**, and
# MAGIC pick `deploy/dataset.csv` from the repo (~7 MB).
# MAGIC
# MAGIC Then re-run the cell below to confirm.

# COMMAND ----------

raw_csv = f"{P.volume('raw')}/dataset.csv"
try:
    n = spark.read.option("header", True).csv(raw_csv).count()
    print(f"OK  {raw_csv}  ({n:,} rows)")
except Exception as e:
    print(f"NOT FOUND  {raw_csv}\nUpload dataset.csv to the raw volume, then re-run.\n\n{e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conventions
# MAGIC
# MAGIC | layer | table | contents |
# MAGIC |---|---|---|
# MAGIC | Bronze | `bronze_invoice_events` | raw events exactly as received, plus ingestion metadata |
# MAGIC | Silver | `silver_invoices` | deduped, validated invoice records |
# MAGIC | Silver | `silver_payments` | deduped, validated payment records |
# MAGIC | Silver | `silver_invoice_outcomes` | invoices joined to payments — carries `days_late` label |
# MAGIC | Silver | `silver_quarantine` | rows that failed validation, with a reason |
# MAGIC | Silver | `silver_data_quality_metrics` | one row per pipeline run per stage |
# MAGIC | Gold | `gold_customer_payment_behavior` | descriptive per-customer aggregates (dashboards) |
# MAGIC | Feature | `feat_customer_payment_history` | **point-in-time** time-series feature table |
# MAGIC
# MAGIC The Gold and Feature tables look similar but are not interchangeable. Gold is a
# MAGIC full-history snapshot for reporting. The feature table is time-series keyed on
# MAGIC `(customer_id, clear_date)` so model lookups can only see already-cleared outcomes.
# MAGIC Feeding Gold to training would reintroduce the exact leak this project fixed.

# COMMAND ----------

for key in TABLES:
    print(f"{key:<18} {P.table(key)}")
