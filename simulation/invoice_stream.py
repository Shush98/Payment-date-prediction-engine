# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic invoice event generator (Phase 2)
# MAGIC
# MAGIC Replays the real dataset as a stream of JSON events into the landing volume, where
# MAGIC Auto Loader picks them up.
# MAGIC
# MAGIC **Two event types, deliberately.** An invoice is raised on `posting_date`; the matching
# MAGIC payment arrives days or weeks later. Emitting them separately is what makes the delayed
# MAGIC label real — at the moment `invoice_created` lands, the outcome genuinely does not exist yet.
# MAGIC A single flat event carrying `clear_date` would smuggle the answer in with the question.
# MAGIC
# MAGIC **This is a demo mechanism, not live data.** Say so in the README and the demo.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "payment_ops")
# Defaults cover the whole 50k source. Invoices are emitted in posting-date order and
# the ~10k OPEN ones are the most recent, so a short run emits only closed invoices and
# leaves inference with nothing to score.
dbutils.widgets.text("batches", "10")
dbutils.widgets.text("batch_size", "5000")
dbutils.widgets.text("sleep_seconds", "1")
dbutils.widgets.text("duplicate_rate", "0.02")
dbutils.widgets.text("corrupt_rate", "0.01")
dbutils.widgets.text("reset", "false")

# COMMAND ----------

# Picks up edits to config.py after a git pull.
%load_ext autoreload
%autoreload 2

# COMMAND ----------

import json, os, sys, time, uuid, hashlib, random

# config.py lives in ../databricks relative to this notebook.
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "databricks")))

from config import Paths

P = Paths(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
BATCHES = int(dbutils.widgets.get("batches"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
SLEEP = float(dbutils.widgets.get("sleep_seconds"))
DUPLICATE_RATE = float(dbutils.widgets.get("duplicate_rate"))
CORRUPT_RATE = float(dbutils.widgets.get("corrupt_rate"))
LANDING = P.volume("landing")
random.seed(42)
print("landing:", LANDING)

# COMMAND ----------

if dbutils.widgets.get("reset").lower() == "true":
    # Clears landing files only. Checkpoints must be cleared too or Auto Loader will
    # consider the re-dropped files already processed.
    dbutils.fs.rm(LANDING, True)
    dbutils.fs.mkdirs(LANDING)
    dbutils.fs.rm(P.checkpoint("bronze"), True)
    print("landing + bronze checkpoint cleared")

# COMMAND ----------

import pandas as pd

src = pd.read_csv(f"{P.volume('raw')}/dataset.csv")
src["posting_date"] = pd.to_datetime(src["posting_date"], format="mixed")
src["due_in_date"] = pd.to_datetime(src["due_in_date"].astype(str), format="%Y%m%d")
src["clear_date"] = pd.to_datetime(src["clear_date"], format="mixed", errors="coerce")
src = src.sort_values("posting_date").reset_index(drop=True)
print(f"{len(src):,} source invoices  ({src.isOpen.eq(1).sum():,} open)")

# COMMAND ----------

def _event_id(kind, invoice_id):
    # Deterministic, so a replayed event carries the same id and Silver's dedupe
    # can collapse it. A random uuid here would make duplicates undetectable.
    return hashlib.sha1(f"{kind}:{invoice_id}".encode()).hexdigest()[:24]


def _date(v):
    return None if pd.isna(v) else v.strftime("%Y-%m-%d")


def invoice_created(row):
    return {
        "_event_id": _event_id("created", row.invoice_id),
        "_source": "invoice_stream_sim",
        "event_type": "invoice_created",
        "event_timestamp": pd.Timestamp.utcnow().isoformat(),
        "invoice_id": str(row.invoice_id),
        "customer_id": str(row.cust_number),
        "customer_name": str(row.name_customer),
        "invoice_amount": float(row.total_open_amount),
        "posting_date": _date(row.posting_date),
        "due_date": _date(row.due_in_date),
        "payment_terms": str(row.cust_payment_terms),
        "business_code": str(row.business_code),
        "clear_date": None,
    }


def invoice_paid(row):
    return {
        "_event_id": _event_id("paid", row.invoice_id),
        "_source": "invoice_stream_sim",
        "event_type": "invoice_paid",
        "event_timestamp": pd.Timestamp.utcnow().isoformat(),
        "invoice_id": str(row.invoice_id),
        "customer_id": str(row.cust_number),
        "customer_name": None,
        "invoice_amount": None,
        "posting_date": None,
        "due_date": None,
        "payment_terms": None,
        "business_code": None,
        "clear_date": _date(row.clear_date),
    }


def corrupt(ev):
    """Inject a realistic defect so the Silver quality checks have something to catch."""
    ev = dict(ev)
    which = random.choice(["bad_date", "null_amount", "due_before_posting", "null_id"])
    if which == "bad_date":
        ev["posting_date"] = "31/02/2020"
    elif which == "null_amount":
        ev["invoice_amount"] = None
    elif which == "due_before_posting":
        ev["due_date"] = "2015-01-01"
    else:
        ev["invoice_id"] = None
    ev["_event_id"] = ev["_event_id"] + "-bad"
    return ev

# COMMAND ----------

# The generator is stateless - every run starts from row 0 of the source. Re-running it
# re-emits the same invoices with the same deterministic _event_id, which Silver dedupes.
covered = min(BATCHES * BATCH_SIZE, len(src))
open_in_range = int(src.iloc[:covered].isOpen.eq(1).sum())
print(f"will emit {covered:,} of {len(src):,} invoices "
      f"({open_in_range:,} open, {covered - open_in_range:,} closed)")
if open_in_range == 0:
    print("\nWARNING: no OPEN invoices in this range - 05_batch_inference will have nothing "
          f"to score.\nOpen invoices are the most recent; raise batches x batch_size above "
          f"~{int(src.isOpen.eq(0).sum()):,} to reach them.")

# COMMAND ----------

cursor = 0
for b in range(BATCHES):
    chunk = src.iloc[cursor:cursor + BATCH_SIZE]
    if chunk.empty:
        print("source exhausted")
        break
    cursor += BATCH_SIZE

    events = []
    for row in chunk.itertuples():
        ev = invoice_created(row)
        events.append(corrupt(ev) if random.random() < CORRUPT_RATE else ev)
        # Payment event only for invoices that actually cleared.
        if not pd.isna(row.clear_date):
            events.append(invoice_paid(row))

    # Replay a few events verbatim - at-least-once delivery is normal, and Silver
    # must collapse them rather than double-count.
    dupes = [dict(e) for e in random.sample(events, k=max(1, int(len(events) * DUPLICATE_RATE)))]
    events.extend(dupes)
    random.shuffle(events)

    path = f"{LANDING}/events_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.json"
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    print(f"batch {b+1}/{BATCHES}: {len(events):>5} events ({len(dupes)} replayed) -> {os.path.basename(path)}")
    if b < BATCHES - 1:
        time.sleep(SLEEP)

emitted = src.iloc[:min(cursor, len(src))]
print(f"\ndone. {len(emitted):,} source invoices emitted"
      f"  ({int(emitted.isOpen.eq(1).sum()):,} open / {int(emitted.isOpen.eq(0).sum()):,} closed)")
print(f"  posting dates: {emitted.posting_date.min().date()} .. {emitted.posting_date.max().date()}")
assert emitted.isOpen.eq(1).sum() > 0, (
    "no open invoices emitted - raise batches x batch_size so the run reaches the "
    "most recent invoices, otherwise there is nothing for inference to score")

# COMMAND ----------

display(dbutils.fs.ls(LANDING))
