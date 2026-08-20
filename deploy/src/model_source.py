"""Resolves where the scoring model comes from: Unity Catalog, or local artifacts.

Primary path is the model registered by databricks/04_model_training.py, loaded from the
Unity Catalog registry at startup. That keeps the deployed app and the lakehouse on the
same model version, with real lineage.

Fallback is the joblib trio in artifacts/. It exists because this app is a public
portfolio demo: if the workspace is unreachable, over its daily quota, or the token has
expired, serving slightly staler predictions beats serving a 500. Which path is live is
reported by /health rather than hidden.

Column naming differs between the two pipelines. The registered model was trained on the
Databricks schema and encodes its own categoricals, so the only translation needed is
cust_payment_terms -> payment_terms. day_of_week already matches: the Spark pipeline was
corrected to pandas' 0=Monday convention precisely so a model could cross this boundary.
"""

import os
import threading
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Bound how long MLflow will sit on a network call. Without this a slow or unreachable
# workspace can block for minutes. Set before mlflow is imported anywhere.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")

# What the Unity Catalog model expects. Must stay in step with
# databricks/modeling.py RAW_INPUT_COLS - test_model_source.py asserts it.
UC_INPUT_COLS = [
    "amount_log", "month", "day_of_week", "is_month_end", "is_year_end",
    "cust_avg_days_late", "cust_std_days_late", "cust_invoice_count",
    "cust_min_late", "cust_max_late", "cust_avg_amount",
    "payment_terms", "business_code",
]

# deploy column -> databricks column
COLUMN_ALIASES = {"cust_payment_terms": "payment_terms"}

DEFAULT_MODEL_NAME = "workspace.payment_ops.payment_lateness"
DEFAULT_ALIAS = "champion"


@dataclass
class Predictor:
    """A scoring function plus enough provenance to display honestly."""

    source: str                      # "unity_catalog" | "local_artifacts"
    version: str
    score: Callable[[pd.DataFrame], pd.DataFrame] = field(repr=False)
    detail: str = ""
    # Why the Unity Catalog path was not used. Exposed on /health so diagnosing a
    # fallback does not mean digging through the host's log tab.
    fallback_reason: str = ""

    def as_dict(self):
        out = {"source": self.source, "version": self.version, "detail": self.detail}
        if self.fallback_reason:
            out["fallback_reason"] = self.fallback_reason
        return out


def _uc_payload(enriched: pd.DataFrame) -> pd.DataFrame:
    payload = enriched.rename(columns=COLUMN_ALIASES)
    missing = [c for c in UC_INPUT_COLS if c not in payload.columns]
    if missing:
        raise KeyError(f"cannot build Unity Catalog payload, missing: {missing}")
    return payload[UC_INPUT_COLS]


def _load_unity_catalog(model_name: str, alias: str):
    """Returns (Predictor, "") on success, or (None, reason) so the caller can fall back
    cleanly AND report why. Never raises - a public demo must not 500 on startup."""
    missing = [v for v in ("DATABRICKS_HOST", "DATABRICKS_TOKEN") if not os.getenv(v)]
    if missing:
        return None, f"{' and '.join(missing)} not set"
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # BOTH are required. set_registry_uri alone is not enough: MLflow resolves the
        # registry through the tracking URI, so without set_tracking_uri("databricks")
        # it falls back to a local store and fails with
        # UnsupportedModelRegistryStoreURIException on a sqlite:// path.
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")

        uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(uri)

        try:
            version = MlflowClient().get_model_version_by_alias(model_name, alias).version
        except Exception:
            version = "unknown"

        def score(enriched: pd.DataFrame) -> pd.DataFrame:
            out = model.predict(_uc_payload(enriched))
            return pd.DataFrame({
                "days_late_pred": np.asarray(out["days_late_pred"], dtype=float),
                "days_late_lower": np.asarray(out["days_late_lower"], dtype=float),
                "days_late_upper": np.asarray(out["days_late_upper"], dtype=float),
            }, index=enriched.index)

        return Predictor("unity_catalog", str(version), score, detail=uri), ""
    except Exception as e:                      # noqa: BLE001 - any failure must fall back
        reason = f"{type(e).__name__}: {e}"
        warnings.warn(f"Unity Catalog model unavailable ({reason}); "
                      "falling back to local artifacts")
        return None, reason


def _load_local(models: dict) -> Predictor:
    from src.features import FEATURE_COLS

    def score(enriched: pd.DataFrame) -> pd.DataFrame:
        X = enriched[FEATURE_COLS].values
        return pd.DataFrame({
            "days_late_pred": models["model_mean"].predict(X),
            "days_late_lower": models["model_lower"].predict(X),
            "days_late_upper": models["model_upper"].predict(X),
        }, index=enriched.index)

    return Predictor("local_artifacts", "bundled", score, detail="artifacts/*.joblib")


def load_predictor(models: dict,
                   model_name: str = None,
                   alias: str = None) -> Predictor:
    """Unity Catalog first, local artifacts if it is not reachable."""
    model_name = model_name or os.getenv("UC_MODEL_NAME", DEFAULT_MODEL_NAME)
    alias = alias or os.getenv("UC_MODEL_ALIAS", DEFAULT_ALIAS)

    uc, reason = _load_unity_catalog(model_name, alias)
    if uc is not None:
        return uc

    local = _load_local(models)
    local.fallback_reason = f"{model_name}@{alias} -> {reason}"
    return local


class DeferredPredictor:
    """Serves local artifacts immediately, upgrades to Unity Catalog in the background.

    Loading a registered model is a network call that can take tens of seconds, or hang
    if the workspace is unreachable. Doing that at import time blocks gunicorn's worker
    boot, so the port never opens and the host kills the deploy - which is exactly what
    happened once already.

    So: bind the port first, serve from bundled artifacts, and swap the real model in
    when it arrives. /health reports whichever is currently live.
    """

    def __init__(self, initial: Predictor):
        self._predictor = initial
        self._lock = threading.Lock()

    @property
    def source(self):
        return self._predictor.source

    @property
    def version(self):
        return self._predictor.version

    @property
    def detail(self):
        return self._predictor.detail

    def score(self, enriched: pd.DataFrame) -> pd.DataFrame:
        with self._lock:
            current = self._predictor
        return current.score(enriched)

    def as_dict(self):
        with self._lock:
            return self._predictor.as_dict()

    def _swap(self, predictor: Predictor):
        with self._lock:
            self._predictor = predictor


def load_predictor_deferred(models: dict,
                            model_name: str = None,
                            alias: str = None) -> DeferredPredictor:
    """Non-blocking. Returns immediately with the local model; upgrades asynchronously."""
    model_name = model_name or os.getenv("UC_MODEL_NAME", DEFAULT_MODEL_NAME)
    alias = alias or os.getenv("UC_MODEL_ALIAS", DEFAULT_ALIAS)

    local = _load_local(models)
    local.fallback_reason = f"{model_name}@{alias} -> loading in background"
    handle = DeferredPredictor(local)

    def _upgrade():
        uc, reason = _load_unity_catalog(model_name, alias)
        if uc is not None:
            handle._swap(uc)
            print(f"[model] upgraded to {uc.source} ({uc.version})")
        else:
            local.fallback_reason = f"{model_name}@{alias} -> {reason}"
            print(f"[model] staying on local artifacts: {reason}")

    threading.Thread(target=_upgrade, name="uc-model-load", daemon=True).start()
    return handle
