import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.features import add_features, FEATURE_COLS

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"


def load_models():
    """Load trained models and encoders from the artifacts directory."""
    return {
        "model_mean":       joblib.load(ARTIFACTS_DIR / "model_mean.joblib"),
        "model_lower":      joblib.load(ARTIFACTS_DIR / "model_lower.joblib"),
        "model_upper":      joblib.load(ARTIFACTS_DIR / "model_upper.joblib"),
        "label_encoders":    joblib.load(ARTIFACTS_DIR / "label_encoders.joblib"),
        "customer_timeline": joblib.load(ARTIFACTS_DIR / "customer_timeline.joblib"),
        "customer_history":  joblib.load(ARTIFACTS_DIR / "customer_history.joblib"),
        "global_defaults":   joblib.load(ARTIFACTS_DIR / "global_defaults.joblib"),
    }


def predict(df, models, predictor=None):
    """Run payment date predictions on a DataFrame of open invoices.

    Features are always built locally from the point-in-time customer timeline. Only the
    scoring step varies: `predictor` may be backed by the Unity Catalog model or by the
    bundled joblib artifacts (see src/model_source.py). Passing None uses the local one.

    Adds four prediction columns to the returned DataFrame:
        days_late_pred  — point estimate (mean model)
        days_late_lower — 10th-percentile estimate (optimistic)
        days_late_upper — 90th-percentile estimate (pessimistic)
        predicted_payment_date — due_in_date + days_late_pred
    """
    enriched = add_features(
        df,
        models["label_encoders"],
        models["customer_timeline"],
        models["global_defaults"],
    )

    if predictor is None:
        from src.model_source import _load_local
        predictor = _load_local(models)

    scored = predictor.score(enriched)
    enriched["days_late_pred"] = scored["days_late_pred"].to_numpy().round(1)
    enriched["days_late_lower"] = scored["days_late_lower"].to_numpy().round(1)
    enriched["days_late_upper"] = scored["days_late_upper"].to_numpy().round(1)

    enriched["predicted_payment_date"] = (
        pd.to_datetime(enriched["due_in_date"])
        + pd.to_timedelta(np.ceil(enriched["days_late_pred"]).astype(int), unit="D")
    )

    return enriched
