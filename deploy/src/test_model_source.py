"""Checks the Flask app can actually feed the Unity Catalog model.

The registered model was trained on the Databricks column schema. If these two drift,
the app would either crash at startup or - far worse - silently pass the wrong columns
and serve confident nonsense.

Run: python -m src.test_model_source      (from the deploy/ directory)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "databricks"))

from src.model_source import (COLUMN_ALIASES, UC_INPUT_COLS, Predictor, _load_local,
                              _uc_payload, load_predictor)


class _PickleStub:
    """Module level so joblib can pickle it into fixture files."""

    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full(len(X), self.v, dtype=float)


def test_uc_input_cols_match_databricks():
    """The single source of truth is databricks/modeling.py RAW_INPUT_COLS."""
    from modeling import RAW_INPUT_COLS
    assert sorted(UC_INPUT_COLS) == sorted(RAW_INPUT_COLS), (
        f"\n  deploy:     {sorted(UC_INPUT_COLS)}\n  databricks: {sorted(RAW_INPUT_COLS)}")


def test_payload_translates_deploy_schema():
    """add_features() output uses cust_payment_terms; the model wants payment_terms."""
    enriched = pd.DataFrame([{
        "amount_log": 9.2, "month": 3, "day_of_week": 1, "is_month_end": 0, "is_year_end": 0,
        "cust_avg_days_late": 2.0, "cust_std_days_late": 5.0, "cust_invoice_count": 4.0,
        "cust_min_late": -3.0, "cust_max_late": 12.0, "cust_avg_amount": 40000.0,
        "cust_payment_terms": "NAA8", "business_code": "U001",
        "total_open_amount": 10000.0,          # extra app columns must be dropped
    }])
    payload = _uc_payload(enriched)
    assert list(payload.columns) == UC_INPUT_COLS, "column order must match the signature"
    assert payload.payment_terms.iloc[0] == "NAA8"
    assert "total_open_amount" not in payload.columns


def test_payload_fails_loudly_on_missing_column():
    bad = pd.DataFrame([{"amount_log": 1.0}])
    try:
        _uc_payload(bad)
    except KeyError as e:
        assert "missing" in str(e)
        return
    raise AssertionError("missing columns must raise, not silently pass a short frame")


def test_aliases_only_rename_what_differs():
    """Every alias target must be something the model actually asks for."""
    for src_col, dst_col in COLUMN_ALIASES.items():
        assert dst_col in UC_INPUT_COLS, f"alias {src_col}->{dst_col} targets an unused column"
        assert src_col not in UC_INPUT_COLS, f"{src_col} needs no alias"


def test_falls_back_to_local_without_credentials(monkeypatch_env=None):
    """No DATABRICKS_HOST/TOKEN -> local artifacts, no exception."""
    import os
    saved = {k: os.environ.pop(k, None) for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN")}
    try:
        class _Stub:
            def predict(self, X): return np.zeros(len(X))
        models = {"model_mean": _Stub(), "model_lower": _Stub(), "model_upper": _Stub()}
        p = load_predictor(models)
        assert p.source == "local_artifacts", p.source
        assert p.version == "bundled"
        # The reason must name the missing variables, so /health explains itself.
        assert "DATABRICKS_HOST" in p.fallback_reason, p.fallback_reason
        assert "DATABRICKS_TOKEN" in p.fallback_reason, p.fallback_reason
        assert "fallback_reason" in p.as_dict()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_local_scorer_returns_expected_shape():
    from src.features import FEATURE_COLS

    class _Stub:
        def __init__(self, v): self.v = v
        def predict(self, X): return np.full(len(X), self.v, dtype=float)

    models = {"model_mean": _Stub(4.0), "model_lower": _Stub(1.0), "model_upper": _Stub(9.0)}
    enriched = pd.DataFrame([{c: 0.0 for c in FEATURE_COLS} for _ in range(3)], index=[7, 8, 9])
    out = _load_local(models).score(enriched)
    assert list(out.columns) == ["days_late_pred", "days_late_lower", "days_late_upper"]
    assert out.index.tolist() == [7, 8, 9], "index must survive so columns align on assignment"
    assert out.days_late_pred.tolist() == [4.0, 4.0, 4.0]


def test_uc_sets_both_tracking_and_registry_uris():
    """Regression: setting only the registry URI leaves MLflow resolving through the
    default tracking store, which failed in production with
    UnsupportedModelRegistryStoreURIException on a sqlite:// path.

    mlflow is not installed locally, so a stub records the call order instead.
    """
    import os
    import types

    calls = []

    class _Loaded:
        def predict(self, df):
            n = len(df)
            return pd.DataFrame({"days_late_pred": [1.0] * n,
                                 "days_late_lower": [0.0] * n,
                                 "days_late_upper": [2.0] * n})

    def _load_model(uri):
        calls.append(("load", uri))
        return _Loaded()

    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = lambda u: calls.append(("tracking", u))
    fake.set_registry_uri = lambda u: calls.append(("registry", u))
    fake.pyfunc = types.SimpleNamespace(load_model=_load_model)

    tracking_mod = types.ModuleType("mlflow.tracking")
    tracking_mod.MlflowClient = lambda: types.SimpleNamespace(
        get_model_version_by_alias=lambda n, a: types.SimpleNamespace(version="7"))
    fake.tracking = tracking_mod

    saved_mods = {k: sys.modules.get(k) for k in ("mlflow", "mlflow.tracking")}
    saved_env = {k: os.environ.get(k) for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN")}
    sys.modules["mlflow"], sys.modules["mlflow.tracking"] = fake, tracking_mod
    os.environ["DATABRICKS_HOST"], os.environ["DATABRICKS_TOKEN"] = "https://x", "tok"
    try:
        from src.model_source import _load_unity_catalog
        pred, reason = _load_unity_catalog("cat.sch.payment_lateness", "champion")

        assert reason == "", f"unexpected failure: {reason}"
        assert pred is not None and pred.source == "unity_catalog"
        assert pred.version == "7"

        assert ("tracking", "databricks") in calls, "tracking URI never set"
        assert ("registry", "databricks-uc") in calls, "registry URI never set"
        load_at = next(i for i, c in enumerate(calls) if c[0] == "load")
        assert calls.index(("tracking", "databricks")) < load_at, "tracking URI set too late"
        assert calls.index(("registry", "databricks-uc")) < load_at, "registry URI set too late"
        assert calls[load_at][1] == "models:/cat.sch.payment_lateness@champion"
    finally:
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_deferred_loader_returns_immediately():
    """Regression: loading the UC model inline blocked gunicorn's worker boot, so the
    port never opened and Render killed the deploy with 'No open ports detected'.

    The loader must return fast even when the Unity Catalog path hangs.
    """
    import os
    import time
    import types

    from src.model_source import load_predictor_deferred

    # A stub whose load_model sleeps, standing in for an unreachable workspace.
    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = lambda u: None
    fake.set_registry_uri = lambda u: None
    fake.pyfunc = types.SimpleNamespace(load_model=lambda uri: time.sleep(5))
    tracking_mod = types.ModuleType("mlflow.tracking")
    tracking_mod.MlflowClient = lambda: types.SimpleNamespace(
        get_model_version_by_alias=lambda n, a: types.SimpleNamespace(version="1"))
    fake.tracking = tracking_mod

    saved_mods = {k: sys.modules.get(k) for k in ("mlflow", "mlflow.tracking")}
    saved_env = {k: os.environ.get(k) for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN")}
    sys.modules["mlflow"], sys.modules["mlflow.tracking"] = fake, tracking_mod
    os.environ["DATABRICKS_HOST"], os.environ["DATABRICKS_TOKEN"] = "https://x", "tok"
    try:
        class _S:
            def predict(self, X): return np.zeros(len(X))
        models = {"model_mean": _S(), "model_lower": _S(), "model_upper": _S()}

        start = time.monotonic()
        handle = load_predictor_deferred(models)
        handle.ensure_started()
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"loader blocked for {elapsed:.1f}s - would stall worker boot"
        assert handle.source == "local_artifacts", "must serve something immediately"
        reason = handle.as_dict()["fallback_reason"]
        assert "running" in reason and "attempt 1/3" in reason, reason

        # And it must be usable right away, not just constructed.
        from src.features import FEATURE_COLS
        enriched = pd.DataFrame([{c: 0.0 for c in FEATURE_COLS}])
        assert len(handle.score(enriched)) == 1
    finally:
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_load_is_not_started_at_construction():
    """Regression: a thread started during module import does not survive gunicorn's
    fork, leaving the serving process stuck on 'loading' forever. Nothing may start
    until a request calls ensure_started()."""
    from src.model_source import DeferredPredictor, Predictor

    started = []
    handle = DeferredPredictor(Predictor("local_artifacts", "bundled", lambda df: None),
                               loader=lambda: (started.append(1), (None, "no"))[1],
                               label="cat.sch.m@champion")
    assert handle._thread is None, "must not spawn a thread at construction"
    assert started == [], "loader must not run until ensure_started()"
    assert "not started yet" in handle.as_dict()["fallback_reason"]


def test_ensure_started_revives_a_vanished_thread():
    """If the thread disappears (e.g. it was created pre-fork), the next request must
    restart it rather than report 'loading' indefinitely."""
    import time as _t
    from src.model_source import DeferredPredictor, Predictor

    runs = []
    handle = DeferredPredictor(Predictor("local_artifacts", "bundled", lambda df: None),
                               loader=lambda: (runs.append(1), (None, "nope"))[1],
                               label="cat.sch.m@champion")
    handle.ensure_started()
    for _ in range(50):                     # let the first attempt finish
        if runs:
            break
        _t.sleep(0.02)
    assert runs == [1], runs

    # It failed terminally, so further calls must not respawn.
    handle.ensure_started()
    assert len(runs) == 1, "a terminal failure must not be retried forever"
    assert "nope" in handle.as_dict()["fallback_reason"]


def test_hung_load_goes_terminal_after_timeout():
    """A load still running past the deadline must stop reporting progress. Python
    cannot kill the thread, but /health must not imply work that is not happening."""
    import threading as _th
    import time as _t
    from src.model_source import DeferredPredictor, Predictor

    stop = _th.Event()
    handle = DeferredPredictor(Predictor("local_artifacts", "bundled", lambda df: None),
                               loader=lambda: (stop.wait(30), (None, "never"))[1],
                               label="m@champion")
    handle.ensure_started()
    try:
        for _ in range(50):
            if handle._thread and handle._thread.is_alive():
                break
            _t.sleep(0.02)
        # Pretend the deadline has passed.
        handle._started = _t.monotonic() - (DeferredPredictor.LOAD_TIMEOUT_SECONDS + 5)
        reason = handle.as_dict()["fallback_reason"]
        assert "timed out after" in reason, reason
        assert "DATABRICKS_AUTH_TYPE=pat" in reason, "must say how to fix it"
        assert handle._state == "failed"
    finally:
        stop.set()


def test_auth_type_is_pinned_when_token_present():
    """Regression: the SDK credential chain probes instance metadata and hangs."""
    import os
    from src.model_source import _load_unity_catalog

    saved = {k: os.environ.get(k) for k in
             ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_AUTH_TYPE")}
    os.environ["DATABRICKS_HOST"] = "https://x"
    os.environ["DATABRICKS_TOKEN"] = "tok"
    os.environ.pop("DATABRICKS_AUTH_TYPE", None)
    try:
        _load_unity_catalog("cat.sch.m", "champion")     # mlflow absent -> returns a reason
        assert os.environ.get("DATABRICKS_AUTH_TYPE") == "pat"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_retry_attempts_are_capped():
    from src.model_source import DeferredPredictor, Predictor

    handle = DeferredPredictor(Predictor("local_artifacts", "bundled", lambda df: None),
                               loader=lambda: (None, "x"), label="m@champion")
    handle._attempts = DeferredPredictor.MAX_ATTEMPTS
    handle.ensure_started()
    assert "gave up after" in handle.as_dict()["fallback_reason"]


def test_failed_load_records_terminal_reason():
    from src.model_source import DeferredPredictor, Predictor

    handle = DeferredPredictor(Predictor("local_artifacts", "bundled", lambda df: None))
    handle._fail("cat.sch.m@champion -> PermissionError: 403 (after 2.1s)")
    reason = handle.as_dict()["fallback_reason"]
    assert "403" in reason and "loading" not in reason, reason


def test_deferred_predictor_swaps_under_lock():
    from src.model_source import DeferredPredictor, Predictor

    local = Predictor("local_artifacts", "bundled", lambda df: "LOCAL")
    handle = DeferredPredictor(local)
    assert handle.score(None) == "LOCAL"
    assert handle.source == "local_artifacts"

    handle._swap(Predictor("unity_catalog", "9", lambda df: "UC", detail="models:/x@champion"))
    assert handle.score(None) == "UC", "swap did not take effect"
    assert handle.source == "unity_catalog"
    assert handle.version == "9"
    assert handle.as_dict()["source"] == "unity_catalog"


def test_baked_model_is_used_and_needs_no_network():
    """The build-time model must be served directly: no mlflow import, no thread, no
    network. That is the whole point of fetching at build time."""
    import json
    import shutil
    import tempfile
    from pathlib import Path as _P

    import joblib

    import src.model_source as ms
    from src.features import FEATURE_COLS

    tmp = _P(tempfile.mkdtemp())
    saved_dir = ms.BAKED_DIR
    try:
        # _PickleStub is module level: joblib cannot pickle a class defined in a function.
        joblib.dump(_PickleStub(4.0), tmp / "model_mean.joblib")
        joblib.dump(_PickleStub(9.0), tmp / "model_lower.joblib")   # crossed on purpose
        joblib.dump(_PickleStub(1.0), tmp / "model_upper.joblib")
        joblib.dump({"payment_terms": {"NAA8": 3}, "business_code": {"U001": 0}},
                    tmp / "encoders.joblib")
        (tmp / "uc_model.json").write_text(json.dumps({
            "model_name": "cat.sch.payment_lateness", "alias": "champion",
            "version": "5", "run_id": "abc", "feature_cols": list(FEATURE_COLS),
            "fetched_at": "2026-08-21T00:00:00Z"}))
        ms.BAKED_DIR = tmp

        handle = ms.load_predictor_deferred({})       # no local models needed
        assert handle.source == "unity_catalog", handle.as_dict()
        assert handle.version == "5"
        assert "baked at build" in handle.detail
        assert handle._thread is None, "baked model must not spawn a loader thread"
        assert "fallback_reason" not in handle.as_dict()

        enriched = pd.DataFrame([{
            **{c: 0.0 for c in FEATURE_COLS},
            "cust_payment_terms": "NAA8", "business_code": "U001"}])
        out = handle.score(enriched)
        assert out.days_late_pred.iloc[0] == 4.0
        # crossed quantiles must still come back ordered
        assert out.days_late_lower.iloc[0] == 1.0
        assert out.days_late_upper.iloc[0] == 9.0
    finally:
        ms.BAKED_DIR = saved_dir
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_baked_model_falls_through_with_reason():
    import tempfile
    from pathlib import Path as _P

    import src.model_source as ms

    saved_dir = ms.BAKED_DIR
    try:
        ms.BAKED_DIR = _P(tempfile.mkdtemp())          # empty
        class _S:
            def predict(self, X): return np.zeros(len(X))
        handle = ms.load_predictor_deferred(
            {"model_mean": _S(), "model_lower": _S(), "model_upper": _S()})
        assert handle.source == "local_artifacts"
        # No report file at all means the build step never executed - that is a distinct
        # diagnosis from a fetch that ran and failed, and the message must say so.
        assert "never ran during build" in handle.as_dict()["fallback_reason"]
    finally:
        ms.BAKED_DIR = saved_dir


def test_build_fetch_report_is_surfaced():
    """A fetch that ran and failed must report its actual error, not just 'absent'."""
    import json as _json
    import shutil
    import tempfile
    from pathlib import Path as _P

    import src.model_source as ms

    saved_dir = ms.BAKED_DIR
    tmp = _P(tempfile.mkdtemp())
    try:
        ms.BAKED_DIR = tmp
        (tmp / "fetch_report.json").write_text(_json.dumps({
            "status": "failed", "message": "PermissionError: 403", "at": "2026-08-21T10:00:00Z"}))
        _, reason = ms._load_baked()
        assert "build fetch failed" in reason and "403" in reason, reason
    finally:
        ms.BAKED_DIR = saved_dir
        shutil.rmtree(tmp, ignore_errors=True)


def test_predictor_reports_provenance():
    p = Predictor("unity_catalog", "3", lambda df: df, detail="models:/x@champion")
    assert p.as_dict() == {"source": "unity_catalog", "version": "3",
                           "detail": "models:/x@champion"}


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {str(e).splitlines()[0][:110]}")
    print("\nAll model-source checks passed." if not failed else f"\n{failed} FAILED")
    sys.exit(1 if failed else 0)
