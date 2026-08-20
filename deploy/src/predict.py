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


def predict(df, models):
    """Run payment date predictions on a DataFrame of open invoices.

    Adds three prediction columns to the returned DataFrame:
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

    X = enriched[FEATURE_COLS].values

    enriched["days_late_pred"]  = models["model_mean"].predict(X).round(1)
    enriched["days_late_lower"] = models["model_lower"].predict(X).round(1)
    enriched["days_late_upper"] = models["model_upper"].predict(X).round(1)

    enriched["predicted_payment_date"] = (
        pd.to_datetime(enriched["due_in_date"])
        + pd.to_timedelta(np.ceil(enriched["days_late_pred"]).astype(int), unit="D")
    )

    return enriched
