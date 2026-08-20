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
