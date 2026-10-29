"""Model loading and prediction logic, kept separate from HTTP routes."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import HTTPException

# Resolve paths from this file, not from whichever folder starts Uvicorn.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "house_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load the model once and reuse it across API requests."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run 'python training.py' from the project root first."
        )
    return joblib.load(MODEL_PATH)


def read_metrics() -> dict[str, Any]:
    """Read real evaluation metrics produced by training.py."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "Model metrics not found. Run 'python training.py' from the project root first."
        )
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def predict_one(values: dict[str, float]) -> float:
    """Return the estimated median value in actual USD."""
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Keep the same column order and names used when training the model.
    row = pd.DataFrame([[values[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    prediction_100k = float(model.predict(row)[0])

    # The California Housing target is measured in units of $100,000.
    return round(prediction_100k * 100_000, 2)
