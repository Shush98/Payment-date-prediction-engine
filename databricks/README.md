# Databricks pipeline — Phases 1–4

Medallion pipeline on Databricks Free Edition: streaming ingestion → quality-checked Silver →
Gold aggregates → **point-in-time feature table**.

## Run order

| # | Notebook | Does |
|---|---|---|
| 0 | `00_setup.py` | Creates schema + volumes. Run once. Then upload `deploy/dataset.csv` to the `raw` volume. |
| 1 | `../simulation/invoice_stream.py` | Emits JSON invoice/payment events into the `landing` volume |
| 2 | `01_bronze_stream.py` | Auto Loader → `bronze_invoice_events` |
| 3 | `02_silver_pipeline.py` | Dedupe, validate, quarantine, join outcomes |
| 4 | `03_feature_engineering.py` | Gold + point-in-time feature table + validation |
| 5 | `04_model_training.py` | Chronological split, 3 quantile models, MLflow, UC registry |
| 6 | `05_batch_inference.py` | Scores the open book into an append-only prediction log |
| 7 | `06_decision_engine.py` | EV ranking + capacity constraint → collection queue |
| 8 | `07_monitoring.py` | Delayed-label join, performance after outcomes, PSI drift |

Re-run 1→4 to add more data. Steps 2–4 are idempotent; step 2 adds zero rows if no new files
arrived, which is the checkpoint working.

All notebooks take `catalog` and `schema` widgets, defaulting to `workspace` / `payment_ops`.

## Free Edition constraints this design works around

| Constraint | Consequence |
|---|---|
| Serverless only; exceeding daily quota shuts the workspace down | **No continuous streaming.** `Trigger.AvailableNow` drains and exits |
| Max 5 concurrent job tasks | Notebooks run sequentially, not as a fan-out DAG |
| One active pipeline per type | Plain notebooks + Workflows rather than Declarative Pipelines |
| One `2X-Small` SQL warehouse | Dashboards query pre-aggregated Gold, never raw Bronze |
| **Online Feature Store unsupported** | Feature lookups are offline/batch. Do not claim real-time feature serving |
| Serverless is not the ML Runtime | `databricks-feature-engineering` needs `%pip install`; notebook 03 falls back to plain Delta if unavailable |
| Outbound internet restricted | No external API calls from notebooks |
| Spark Connect only | DataFrame API only — no RDD APIs, no global temp views |

`Trigger.AvailableNow` is still real Structured Streaming: same checkpointing, same
exactly-once file tracking, same restart semantics. It is triggered rather than resident.

## The two-event design

The generator emits `invoice_created` and `invoice_paid` as **separate events**, because the
payment outcome is a delayed label — at the moment an invoice is raised, the outcome genuinely
does not exist. A single flat event carrying `clear_date` would ship the answer with the
question and make the delayed-label story fictional.

This is what makes Phase 8 monitoring possible later: predictions are logged at
`invoice_created`, and scored when the matching `invoice_paid` arrives.

The generator also injects duplicates (~2%) and corrupt rows (~1%) on purpose, so the Silver
quality checks have something real to catch.

## Point-in-time correctness

The core rule, implemented in `transforms.asof_join_history`:

> When predicting an invoice raised on date `t`, only outcomes that **cleared strictly before
> `t`** may be used.

Strictly before — an invoice raised the same day another clears must not see it. This mirrors
`merge_asof(allow_exact_matches=False)` in `deploy/src/features.py`. Dropping the strictness is
a silent same-day leak.

`03_feature_engineering.py` asserts four properties and fails the run if any break:

1. The as-of join does not duplicate invoices
2. No invoice uses history dated on/after its own `posting_date`
3. No invoice sees its own outcome
4. Cold-start customers fall back to global defaults

Gold and the feature table are **not** interchangeable. Gold aggregates a customer's entire
history including future invoices; using it for training reintroduces the leak the audit
measured at 1.14× MAE inflation.

## Local testing

`transforms.py` is plain PySpark with no Databricks dependencies, so the logic can be verified
before deploying:

```
python databricks/test_transforms.py
```

Eight checks, including the four point-in-time properties above and their pandas equivalents in
`deploy/src/test_features.py`. Both must agree — the Spark pipeline and the Flask app are
supposed to compute the same features.

**On Windows** this needs a 64-bit Java and a Python that pyspark supports. Python 3.14 fails
(pyspark 3.5's vendored cloudpickle is incompatible); Python 3.10 works:

```powershell
$env:JAVA_HOME = "C:\PROGRA~1\Java\jre1.8.0_471"      # short path — spaces break the launcher
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$env:PYTHONPATH = "$env:LOCALAPPDATA\Programs\Python\Python314\Lib\site-packages"
$py = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
$env:PYSPARK_PYTHON = $py; $env:PYSPARK_DRIVER_PYTHON = $py
& $py databricks\test_transforms.py
```

A `winutils.exe` warning and a temp-dir cleanup stack trace on shutdown are expected on Windows
and harmless.

## Files

```
databricks/
  config.py                  naming conventions, Paths resolver
  transforms.py              pure PySpark logic (tested locally)
  test_transforms.py         8 local checks, no Databricks needed
  00_setup.py                schema + volumes
  01_bronze_stream.py        Auto Loader ingestion
  02_silver_pipeline.py      quality, dedup, validation, outcome join
  03_feature_engineering.py  Gold + point-in-time feature table
simulation/
  invoice_stream.py          synthetic event generator
```

## Not yet built

Phases 5–10: MLflow training, Unity Catalog model registry, batch/streaming inference, the
decision engine port, monitoring with delayed labels, and the dashboard.

The decision engine (`deploy/src/decision.py`) stays pandas — it ranks a few hundred rows, so
Spark buys nothing.
