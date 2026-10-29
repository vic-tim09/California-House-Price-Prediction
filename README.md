# SmartHouse AI — Housing Value Prediction API

A beginner-friendly machine-learning project that serves California Housing predictions through a REST API built with FastAPI. It includes model training, evaluation metrics, Pydantic validation, single and batch predictions, and automated API tests.

> **Important limitation:** The California Housing dataset target represents median house value for a census block group. The output is an educational model estimate, not the price of an individual home or a professional appraisal.

## Features

- Random Forest regression model trained with scikit-learn.
- Train/test split and evaluation with MAE, RMSE, and R².
- Saved model artifact so the API does not retrain for every request.
- Validated request schemas with helpful field descriptions.
- `POST /predict` for one prediction.
- `POST /predict/batch` for up to 50 predictions.
- `GET /health` and `GET /model-info` endpoints.
- Interactive API documentation and automated tests.

## Project structure

```text
SmartHouse-AI/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI routes
│   ├── schemas.py          # Request/response validation
│   └── model_service.py    # Model loading and prediction logic
├── models/                 # Generated model and metrics (not committed)
├── tests/test_api.py       # API validation tests
├── Explore.py              # Dataset exploration
├── training.py             # Train and evaluate the model
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup (Windows PowerShell)

Use Python 3.11 or 3.12 for a predictable beginner setup.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.12 is not installed, create the environment with your installed supported Python version instead.

## 1. Explore the data (optional)

```powershell
python Explore.py
```

The first run may download the California Housing dataset and requires an internet connection. scikit-learn caches the dataset for later runs.

## 2. Train the model

```powershell
python training.py
```

This creates:

- `models/house_model.joblib` — the trained Random Forest model.
- `models/metrics.json` — actual metrics calculated on the held-out test data.

These generated artifacts are ignored by Git. Every user should run the training command before making predictions. Training requires an internet connection the first time the dataset is downloaded.

## 3. Start the API

From the project root (the folder containing `training.py`):

```powershell
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive Swagger documentation, or `http://127.0.0.1:8000/redoc` for ReDoc.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Project welcome response |
| GET | `/health` | API status and whether the model loaded |
| GET | `/model-info` | Real model evaluation metrics |
| POST | `/predict` | Predict one block group's median value |
| POST | `/predict/batch` | Predict 1–50 records in one request |

### Example request for `POST /predict`

```json
{
  "MedInc": 5.0,
  "HouseAge": 20.0,
  "AveRooms": 5.0,
  "AveBedrms": 1.0,
  "Population": 1000.0,
  "AveOccup": 3.0,
  "Latitude": 34.0,
  "Longitude": -118.0
}
```

The response contains `estimated_median_house_value_usd`, the currency, the target description, and model name. The exact value depends on the trained model and input data.

For `/predict/batch`, wrap a list of records in a `houses` field. The API limits a batch to 50 records to keep requests bounded.

## Run tests

Train the model first, then run:

```powershell
pytest -q
```

The tests cover core routes and invalid request validation. The model-info test allows either success (if metrics exist) or a helpful `503` response (if training has not yet run).

## Metrics explained

- **MAE:** average absolute prediction error, reported in both target units and USD.
- **RMSE:** penalizes larger errors more strongly than MAE.
- **R²:** compares the model with a baseline that predicts the test target mean; it is not a percentage accuracy score.

Metrics are computed from the held-out test split and written to `models/metrics.json`. Do not claim the model is suitable for real estate pricing without further validation.

## GitHub checklist

1. Add the project files to a Git repository.
2. Do not commit `.venv/`, caches, secrets, or generated model files.
3. Include the training command and API example in the README.
4. Add screenshots of `/docs` and successful sample responses after running the app.
5. Push the repository to GitHub.

## Technologies Used
Python
Pandas
Scikit-learn
FastAPI
Uvicorn
Joblib
