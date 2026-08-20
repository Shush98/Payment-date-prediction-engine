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
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

from pathlib import Path

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

# Where src/fetch_model.py writes the model it pulls from Unity Catalog at build time.
BAKED_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "uc"


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

    # Pin the auth method. Given only a token, the Databricks SDK still walks its whole
    # credential chain - OAuth, cloud CLIs, and instance metadata services. On a hosted
    # box the metadata probe has no useful timeout and the load hangs for minutes.
    # We have a PAT; say so and skip the search.
    os.environ.setdefault("DATABRICKS_AUTH_TYPE", "pat")

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


def _encode(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    out = df.copy()
    out["payment_terms_encoded"] = out["payment_terms"].map(encoders["payment_terms"]).fillna(-1).astype(int)
    out["business_code_encoded"] = out["business_code"].map(encoders["business_code"]).fillna(-1).astype(int)
    return out


def _load_baked():
    """The Unity Catalog model downloaded during the build (see src/fetch_model.py).

    Plain joblib, so this costs no network call and does not import mlflow - which on a
    0.1-CPU host is the difference between instant and several minutes.
    """
    import json

    meta_path = BAKED_DIR / "uc_model.json"
    if not meta_path.exists():
        return None, "no build-time model (artifacts/uc/uc_model.json absent)"
    try:
        import joblib
        from src.features import FEATURE_COLS

        meta = json.loads(meta_path.read_text())
        m_mean = joblib.load(BAKED_DIR / "model_mean.joblib")
        m_lower = joblib.load(BAKED_DIR / "model_lower.joblib")
        m_upper = joblib.load(BAKED_DIR / "model_upper.joblib")
        encoders = joblib.load(BAKED_DIR / "encoders.joblib")

        if list(meta.get("feature_cols", FEATURE_COLS)) != list(FEATURE_COLS):
            return None, "baked model feature order differs from the app's FEATURE_COLS"

        def score(enriched: pd.DataFrame) -> pd.DataFrame:
            X = _encode(enriched.rename(columns=COLUMN_ALIASES), encoders)[FEATURE_COLS].to_numpy()
            lower, upper = m_lower.predict(X), m_upper.predict(X)
            return pd.DataFrame({
                "days_late_pred": m_mean.predict(X),
                "days_late_lower": np.minimum(lower, upper),
                "days_late_upper": np.maximum(lower, upper),
            }, index=enriched.index)

        return Predictor("unity_catalog", meta["version"], score,
                         detail=f"{meta['model_name']}@{meta['alias']} "
                                f"baked at build {meta['fetched_at']}"), ""
    except Exception as e:                  # noqa: BLE001
        return None, f"baked model unreadable: {type(e).__name__}: {e}"


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

    MAX_ATTEMPTS = 3
    # A load still running past this is wedged, not slow. Python cannot kill a thread,
    # so the orphan is left to die with the process - but the state goes terminal so
    # /health stops implying progress that is not happening.
    LOAD_TIMEOUT_SECONDS = 90

    def __init__(self, initial: Predictor, loader=None, label=""):
        self._predictor = initial
        self._loader = loader            # () -> (Predictor | None, reason)
        self._label = label
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._state = "loading"          # loading | active | failed
        self._thread = None
        self._attempts = 0

    def ensure_started(self):
        """Start (or restart) the background load. Idempotent, safe to call per request.

        Deliberately NOT called at import. A thread started during module import does not
        survive the fork that gunicorn does when spawning a worker: the object remains but
        the thread is gone, leaving the load permanently 'in progress' in the process that
        actually serves traffic. Starting on first request guarantees the thread lives in
        the right process, and revives it if it ever disappears.
        """
        if self._loader is None:
            return
        with self._lock:
            if self._state != "loading":
                return
            if self._thread is not None and self._thread.is_alive():
                return
            if self._attempts >= self.MAX_ATTEMPTS:
                self._predictor.fallback_reason = (
                    f"{self._label} -> gave up after {self._attempts} attempts")
                self._state = "failed"
                return
            self._attempts += 1
            self._started = time.monotonic()
            self._thread = threading.Thread(target=self._run, name="uc-model-load", daemon=True)
            thread = self._thread
        thread.start()

    def _run(self):
        started = time.monotonic()
        try:
            uc, reason = self._loader()
        except BaseException as e:      # noqa: BLE001 - a dead thread must not look live
            uc, reason = None, f"loader thread crashed: {type(e).__name__}: {e}"

        took = time.monotonic() - started
        if uc is not None:
            self._swap(uc)
            print(f"[model] upgraded to unity_catalog ({uc.version}) in {took:.1f}s")
        else:
            self._fail(f"{self._label} -> {reason} (after {took:.1f}s)")
            print(f"[model] staying on local artifacts after {took:.1f}s: {reason}")

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
            info = self._predictor.as_dict()
            state, started = self._state, self._started

        if state == "loading":
            elapsed = time.monotonic() - started
            alive = self._thread.is_alive() if self._thread else False

            if alive and elapsed > self.LOAD_TIMEOUT_SECONDS:
                self._fail(f"{self._label} -> timed out after {elapsed:.0f}s "
                           "(hung, not slow - usually the Databricks SDK credential "
                           "chain; set DATABRICKS_AUTH_TYPE=pat)")
                return self.as_dict()

            if self._thread is None:
                stage = "not started yet"
            else:
                stage = "running" if alive else "thread gone, will retry"
            status = (f"unity catalog load {stage} for {elapsed:.0f}s "
                      f"(attempt {self._attempts}/{self.MAX_ATTEMPTS})")
            # Keep the original reason. It says why the build-time model was not used,
            # which is the more useful half - overwriting it hid that entirely.
            base = info.get("fallback_reason", "")
            info["fallback_reason"] = f"{base} | {status}" if base else status
        return info

    def _swap(self, predictor: Predictor):
        with self._lock:
            self._predictor = predictor
            self._state = "active"

    def _fail(self, reason: str):
        with self._lock:
            self._predictor.fallback_reason = reason
            self._state = "failed"


def load_predictor_deferred(models: dict,
                            model_name: str = None,
                            alias: str = None) -> DeferredPredictor:
    """Non-blocking. Resolution order:

    1. The model baked in at build time by src/fetch_model.py - instant, no network.
       This is the normal path in a deployed build.
    2. Bundled joblib artifacts, while a live Unity Catalog load is attempted in the
       background on first request. Only used when the build-time fetch did not happen,
       e.g. running locally without credentials.
    """
    model_name = model_name or os.getenv("UC_MODEL_NAME", DEFAULT_MODEL_NAME)
    alias = alias or os.getenv("UC_MODEL_ALIAS", DEFAULT_ALIAS)

    baked, baked_reason = _load_baked()
    if baked is not None:
        handle = DeferredPredictor(baked)
        handle._state = "active"           # nothing to load, nothing to retry
        return handle

    local = _load_local(models)
    local.fallback_reason = baked_reason
    return DeferredPredictor(
        local,
        loader=lambda: _load_unity_catalog(model_name, alias),
        label=f"{model_name}@{alias}",
    )
