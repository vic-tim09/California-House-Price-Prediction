"""API tests. Run with: pytest -q"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_HOUSE = {
    "MedInc": 5.0,
    "HouseAge": 20.0,
    "AveRooms": 5.0,
    "AveBedrms": 1.0,
    "Population": 1000.0,
    "AveOccup": 3.0,
    "Latitude": 34.0,
    "Longitude": -118.0,
}


def test_home_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["project"] == "SmartHouse AI"


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediction_rejects_invalid_input() -> None:
    invalid_house = {**VALID_HOUSE, "MedInc": -1}
    response = client.post("/predict", json=invalid_house)
    assert response.status_code == 422


def test_prediction_rejects_unknown_fields() -> None:
    invalid_house = {**VALID_HOUSE, "MedIncc": 5.0}
    response = client.post("/predict", json=invalid_house)
    assert response.status_code == 422


def test_batch_rejects_empty_list() -> None:
    response = client.post("/predict/batch", json={"houses": []})
    assert response.status_code == 422


def test_model_info_before_training_returns_helpful_error() -> None:
    # If training hasn't run, the API should explain the next step rather than crash.
    response = client.get("/model-info")
    assert response.status_code in (200, 503)
