---
title: B2B Payment Risk & Collection Decision Engine
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# B2B Payment Risk & Collection Decision Engine

Predicts when B2B invoices will be paid, then decides **which invoices a capacity-limited
collections team should call today** — ranked by expected monetary value, not by lateness
or invoice size.

The model is one component. The output is a work order.

## What it does

| Tab | Shows |
|---|---|
| Operations | Open invoices with confidence ranges; the ranked collection queue |
| Impact Analysis | Strategy comparison (random vs by-amount vs decision engine), value at risk |
| Model Intelligence | Feature importance, customer risk segmentation, interval calibration |

Press **Predict** to score the open book. Scoring ~10k invoices takes a few seconds.

## The decision, not the prediction

```
EV = P(late) × P(responds) × days_accelerated × daily_capital_rate × amount − call_cost
```

`P(late)` comes from the **P10/P90 prediction interval**, not the point estimate — so
interval calibration matters more here than MAE. Invoices are tiered CALL / REMIND /
WATCH / OK, and the CALL tier is capped at the team's daily capacity (20); overflow is
demoted to REMIND rather than dropped.

## Model provenance

`GET /health` reports which model is actually serving:

```json
{"status": "ok", "model": {"source": "unity_catalog", "version": "3", ...}}
```

- **`unity_catalog`** — the champion model from the Databricks Unity Catalog registry,
  the same version the lakehouse pipeline uses.
- **`local_artifacts`** — bundled joblib fallback, used when the workspace is
  unreachable. Reported rather than hidden, so a silent fallback never masquerades as
  the live model.

To enable the Unity Catalog path, set these as **Space secrets**
(Settings → Variables and secrets):

| Secret | Value |
|---|---|
| `DATABRICKS_HOST` | `https://<your-workspace>.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | a personal access token |
| `UC_MODEL_NAME` | optional, defaults to `workspace.payment_ops.payment_lateness` |
| `UC_MODEL_ALIAS` | optional, defaults to `champion` |

## Honest limitations

- **Prediction accuracy is ~3.2 days MAE** on a chronological hold-out. An earlier
  version of this project reported 2.69 — that number was target leakage, not skill.
  Customer-history features were aggregated over all invoices including each row's own
  outcome. Fixed with point-in-time features; the honest number is worse and real.
- **Intervention effectiveness is assumed, not measured.** `P(responds)` and
  `days_accelerated` are parameters. No collection calls were ever actually made, so the
  expected-value uplift compares *orderings* under the engine's own assumptions — it is
  not an observed business outcome. Measuring it needs a holdout where some queued
  invoices are deliberately not called.
- **The dataset is historical** (2020 B2B invoices). Streaming in the companion
  Databricks pipeline is a demonstration mechanism, not live data.

## Running locally

```bash
pip install -r requirements.txt
python app.py          # http://localhost:5000
```

Without `DATABRICKS_*` env vars it uses the bundled artifacts — no workspace needed.

## Source

Full project, including the Databricks medallion pipeline, MLflow training, and the
leakage audit: <https://github.com/Shush98/Payment-date-prediction-engine>
