# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze: streaming ingestion (Phase 2)
# MAGIC
# MAGIC Auto Loader reads the JSON event files the generator drops into the landing volume and
# MAGIC appends them to a Delta table, unchanged apart from ingestion metadata.
# MAGIC
# MAGIC **`Trigger.AvailableNow`, not a continuous stream.** On Free Edition an always-on
# MAGIC readStream burns the daily serverless quota and gets the whole workspace shut down until
# MAGIC the next day. `availableNow` drains everything currently waiting and stops. It is still
# MAGIC genuine Structured Streaming — same checkpointing, same exactly-once file tracking, same
# MAGIC restart semantics — just triggered rather than resident.
# MAGIC
# MAGIC **Bronze keeps everything.** No filtering, no casting, no dropping of bad rows. If a
# MAGIC parsing rule turns out to be wrong, the original is still here to reprocess. Cleaning
# MAGIC happens in Silver.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "payment_ops")

# COMMAND ----------

# Picks up edits to transforms.py / config.py after a git pull.
%load_ext autoreload
%autoreload 2

# COMMAND ----------

import os, sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from pyspark.sql import functions as F
from config import Paths
from transforms import INVOICE_EVENT_SCHEMA

P = Paths(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
LANDING = P.volume("landing")
CKPT = P.checkpoint("bronze")
BRONZE = P.table("bronze_events")
print(f"{LANDING}\n  -> {BRONZE}\n  checkpoint: {CKPT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The read stream
# MAGIC
# MAGIC An explicit schema is supplied rather than letting Auto Loader infer it. Inference on a
# MAGIC folder of JSON is fine for exploration, but it makes the pipeline's contract depend on
# MAGIC whichever files happened to arrive first — a new nullable column or an all-null batch can
# MAGIC silently change types between runs.

# COMMAND ----------

stream = (spark.readStream
          .format("cloudFiles")
          .option("cloudFiles.format", "json")
          .option("cloudFiles.schemaLocation", f"{CKPT}/schema")
          .schema(INVOICE_EVENT_SCHEMA)
          .load(LANDING)
          .withColumn("_ingested_at", F.current_timestamp())
          .withColumn("_source_file", F.col("_metadata.file_path"))
          .withColumn("_ingest_date", F.current_date()))

# COMMAND ----------

query = (stream.writeStream
         .format("delta")
         .outputMode("append")
         .option("checkpointLocation", CKPT)
         .option("mergeSchema", "true")
         .trigger(availableNow=True)
         .toTable(BRONZE))

query.awaitTermination()
print("ingestion complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What landed
# MAGIC
# MAGIC Re-running this notebook without new files should add **zero** rows: the checkpoint
# MAGIC records which files were consumed. That is the idempotency guarantee — worth
# MAGIC demonstrating live, because "just run it again" is the most common real-world recovery.

# COMMAND ----------

last = query.lastProgress
if last:
    print("batch id       :", last.get("batchId"))
    print("rows this run  :", last.get("numInputRows"))

bronze = spark.table(BRONZE)
print("total rows in bronze:", bronze.count())
display(bronze.groupBy("event_type").count())

# COMMAND ----------

display(bronze.orderBy(F.col("_ingested_at").desc()).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC Duplicate `_event_id`s are expected here — the generator replays a fraction of events on
# MAGIC purpose, and at-least-once delivery is normal for file-based ingestion. Bronze records
# MAGIC them faithfully; Silver collapses them.

# COMMAND ----------

dupes = (bronze.groupBy("_event_id").count().filter("count > 1"))
print("event_ids arriving more than once:", dupes.count())
display(dupes.orderBy(F.col("count").desc()).limit(10))
