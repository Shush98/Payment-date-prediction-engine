"""Download the Unity Catalog champion model at BUILD time, not at runtime.

Why build time:
  * Render's free tier runs on 0.1 CPU. Importing mlflow there takes minutes, so doing
    it while serving requests meant the model never arrived before the watchdog fired.
    The build step has no such pressure and is allowed to be slow.
  * The registered pyfunc references databricks/modeling.py *by reference* (it was logged
    without code_paths). That module exists in this repo, so it can be unwrapped here -
    but it would not be importable from the deployed app on its own.

So this unwraps the bundle and writes plain joblib files. At runtime the app loads those
directly: no mlflow import, no network call, no Databricks dependency at all - while still
serving the exact model version registered in Unity Catalog.

Run during build:  python -m src.fetch_model
Always exits 0 - a missing model must never fail the deploy, it just falls back.
"""

import json
import os
import sys
import time
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent.parent
BAKED_DIR = DEPLOY_DIR / "artifacts" / "uc"
# modeling.py must be importable to unpickle the bundle it was logged with.
sys.path.insert(0, str(DEPLOY_DIR.parent / "databricks"))

DEFAULT_MODEL_NAME = "workspace.payment_ops.payment_lateness"
DEFAULT_ALIAS = "champion"


def main() -> int:
    model_name = os.getenv("UC_MODEL_NAME", DEFAULT_MODEL_NAME)
    alias = os.getenv("UC_MODEL_ALIAS", DEFAULT_ALIAS)

    if not (os.getenv("DATABRICKS_HOST") and os.getenv("DATABRICKS_TOKEN")):
        print("[fetch_model] DATABRICKS_HOST/DATABRICKS_TOKEN not set - skipping. "
              "The app will serve bundled artifacts.")
        return 0

    os.environ.setdefault("DATABRICKS_AUTH_TYPE", "pat")
    started = time.monotonic()
    try:
        import joblib
        import mlflow
        from mlflow.tracking import MlflowClient

        # Both are required: MLflow resolves the registry through the tracking URI.
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")

        client = MlflowClient()
        mv = client.get_model_version_by_alias(model_name, alias)
        print(f"[fetch_model] {model_name}@{alias} -> version {mv.version}")

        loaded = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
        bundle = loaded.unwrap_python_model().bundle

        BAKED_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle.model_mean, BAKED_DIR / "model_mean.joblib")
        joblib.dump(bundle.model_lower, BAKED_DIR / "model_lower.joblib")
        joblib.dump(bundle.model_upper, BAKED_DIR / "model_upper.joblib")
        joblib.dump(bundle.encoders, BAKED_DIR / "encoders.joblib")

        meta = {
            "model_name": model_name,
            "alias": alias,
            "version": str(mv.version),
            "run_id": mv.run_id,
            "feature_cols": list(bundle.feature_cols),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (BAKED_DIR / "uc_model.json").write_text(json.dumps(meta, indent=2))

        print(f"[fetch_model] baked version {mv.version} into {BAKED_DIR} "
              f"in {time.monotonic() - started:.1f}s")
        return 0

    except Exception as e:                  # noqa: BLE001 - never fail the build
        print(f"[fetch_model] FAILED after {time.monotonic() - started:.1f}s: "
              f"{type(e).__name__}: {e}")
        print("[fetch_model] continuing - the app will serve bundled artifacts.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
