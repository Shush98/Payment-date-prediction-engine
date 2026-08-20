"""Train models and save all artifacts for the web application."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    ARTIFACTS_DIR, DATASET_PATH, FEATURE_COLS,
    MODEL_MEAN_PATH, MODEL_LOWER_PATH, MODEL_UPPER_PATH,
    LABEL_ENCODERS_PATH, CUSTOMER_HISTORY_PATH, GLOBAL_DEFAULTS_PATH,
    TEST_RESULTS_PATH,
)
from app.models.feature_engineering import engineer_features, compute_customer_responsiveness
from app.models.decision_engine import DecisionEngine


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and parse data
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    df["posting_date"] = pd.to_datetime(df["posting_date"], format="mixed")
    df["due_in_date"] = pd.to_datetime(df["due_in_date"].astype(str), format="%Y%m%d")
    df["clear_date"] = pd.to_datetime(df["clear_date"], format="mixed", errors="coerce")

    closed = df[df["isOpen"] == 0].copy()
    closed["days_late"] = (closed["clear_date"] - closed["due_in_date"]).dt.days
    print(f"Closed invoices: {len(closed)}")

    # 2. Feature engineering (training mode — builds encoders and history)
    print("Engineering features...")
    featured, label_encoders, customer_history, global_defaults = engineer_features(closed)

    # 3. Save encoders and history
    print("Saving label encoders, customer history, and global defaults...")
    joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
    joblib.dump(customer_history, CUSTOMER_HISTORY_PATH)
    joblib.dump(global_defaults, GLOBAL_DEFAULTS_PATH)

    # 4. Prepare data and split
    X = featured[FEATURE_COLS].values
    y = featured["days_late"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # 5. Train models
    print("Training mean model...")
    model_mean = GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model_mean.fit(X_train, y_train)

    print("Training lower quantile model (alpha=0.1)...")
    model_lower = GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        loss="quantile", alpha=0.1, random_state=42,
    )
    model_lower.fit(X_train, y_train)

    print("Training upper quantile model (alpha=0.9)...")
    model_upper = GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        loss="quantile", alpha=0.9, random_state=42,
    )
    model_upper.fit(X_train, y_train)

    # 6. Save models
    print("Saving models...")
    joblib.dump(model_mean, MODEL_MEAN_PATH)
    joblib.dump(model_lower, MODEL_LOWER_PATH)
    joblib.dump(model_upper, MODEL_UPPER_PATH)

    # 7. Evaluate
    y_pred = model_mean.predict(X_test)
    y_lower = model_lower.predict(X_test)
    y_upper = model_upper.predict(X_test)

    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    within_3 = float(np.mean(np.abs(y_test - y_pred) <= 3))
    coverage = float(np.mean((y_test >= y_lower) & (y_test <= y_upper)))

    print(f"\n=== Model Performance ===")
    print(f"RMSE: {rmse:.2f} days")
    print(f"MAE: {mae:.2f} days")
    print(f"Within ±3 days: {within_3*100:.1f}%")
    print(f"80% PI Coverage: {coverage*100:.1f}%")

    # 8. Generate test results for the results page
    print("\nGenerating test results...")
    # Get the original test rows (same split as train_test_split)
    _, test_indices = train_test_split(
        range(len(featured)), test_size=0.2, random_state=42
    )
    test_featured = featured.iloc[test_indices].copy()
    test_featured["predicted_days_late"] = y_pred
    test_featured["prediction_lower"] = y_lower
    test_featured["prediction_upper"] = y_upper
    test_featured["actual_days_late"] = y_test
    test_featured["customer_responsiveness"] = compute_customer_responsiveness(test_featured)

    # Run decision engine
    engine = DecisionEngine()
    recommendations = engine.prioritize_invoices(test_featured)

    # Build test results JSON
    invoices_list = []
    for _, row in test_featured.iterrows():
        invoices_list.append({
            "invoice_id": str(row.get("invoice_id", "")),
            "cust_number": str(row.get("cust_number", "")),
            "name_customer": str(row.get("name_customer", "")),
            "invoice_amount": round(float(row["invoice_amount"]), 2),
            "predicted_days_late": round(float(row["predicted_days_late"]), 1),
            "prediction_lower": round(float(row["prediction_lower"]), 1),
            "prediction_upper": round(float(row["prediction_upper"]), 1),
            "actual_days_late": round(float(row["actual_days_late"]), 1),
            "customer_responsiveness": round(float(row["customer_responsiveness"]), 4),
            "cust_std_days_late": round(float(row.get("cust_std_days_late", 10)), 2),
        })

    recommendations_list = recommendations.head(200).to_dict("records")

    test_results = {
        "metrics": {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "within_3_days": round(within_3 * 100, 1),
            "pi_coverage": round(coverage * 100, 1),
        },
        "invoices": invoices_list,
        "recommendations": recommendations_list,
        "summary": {
            "total_invoices": len(test_featured),
            "call_count": int((recommendations["recommendation"] == "CALL").sum()),
            "skip_count": int((recommendations["recommendation"] == "SKIP").sum()),
            "total_expected_value": round(
                float(recommendations[recommendations["expected_value"] > 0]["expected_value"].sum()), 2
            ),
        },
        "impact": {
            "random_ev": round(float(recommendations.sample(20, random_state=42)["expected_value"].sum()), 2),
            "by_amount_ev": round(float(recommendations.nlargest(20, "invoice_amount")["expected_value"].sum()), 2),
            "decision_engine_ev": round(float(recommendations.head(20)["expected_value"].sum()), 2),
        },
    }

    with open(TEST_RESULTS_PATH, "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    print(f"Test results: {len(invoices_list)} invoices")
    print(f"Recommendations: CALL={test_results['summary']['call_count']}, SKIP={test_results['summary']['skip_count']}")
    print(f"Impact: Random=${test_results['impact']['random_ev']}, "
          f"ByAmount=${test_results['impact']['by_amount_ev']}, "
          f"Engine=${test_results['impact']['decision_engine_ev']}")
    print("\nDone!")


if __name__ == "__main__":
    main()
