"""Train and evaluate the SmartHouse AI regression model.

Run from the project root: python training.py
This creates models/house_model.joblib and models/metrics.json.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "house_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
RANDOM_STATE = 42


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # as_frame=True keeps feature names attached to the dataset.
    dataset = fetch_california_housing(as_frame=True)
    X: pd.DataFrame = dataset.data
    y: pd.Series = dataset.target  # Target units are $100,000.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # Random Forest combines many decision trees to improve regression results.
    model = RandomForestRegressor(
        n_estimators=150,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print(f"Training on {len(X_train):,} rows...")
    model.fit(X_train, y_train)

    # Evaluate only on the held-out test set, which the model did not train on.
    predictions = model.predict(X_test)
    mae_100k = mean_absolute_error(y_test, predictions)
    rmse_100k = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "model_name": "RandomForestRegressor",
        "dataset": "California Housing (scikit-learn)",
        "test_rows": int(len(X_test)),
        "mae_100k_usd": round(float(mae_100k), 5),
        "mae_usd": round(float(mae_100k * 100_000), 2),
        "rmse_100k_usd": round(float(rmse_100k), 5),
        "rmse_usd": round(float(rmse_100k * 100_000), 2),
        "r2_score": round(float(r2), 5),
        "target_unit": "$100,000 USD",
        "random_state": RANDOM_STATE,
    }

    # Save the fitted model once so the API doesn't retrain on every request.
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print("\nEvaluation results")
    print(f"MAE:  ${metrics['mae_usd']:,.0f}")
    print(f"RMSE: ${metrics['rmse_usd']:,.0f}")
    print(f"R²:    {metrics['r2_score']}")
    print(f"\nSaved model:   {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
