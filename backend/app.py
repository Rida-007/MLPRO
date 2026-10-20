import sys
import os

# Add the project root to the Python path so we can import the ML package
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import joblib
import pandas as pd
import logging

from backend.schemas import JobPosting, PredictionResponse, HealthResponse
from ml.preprocessing import preprocess_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Model state
ml_models = {}

# Constants
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "tuned",
    "linear_svm_tuned.joblib"
)

THRESHOLD = 0.20


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model on startup
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        ml_models["linear_svm_tuned"] = None
    else:
        try:
            ml_models["linear_svm_tuned"] = joblib.load(MODEL_PATH)
            logger.info("Successfully loaded linear_svm_tuned model")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            ml_models["linear_svm_tuned"] = None

    yield

    # Cleanup on shutdown
    ml_models.clear()


app = FastAPI(
    title="TrueHire Prediction API",
    lifespan=lifespan
)


# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://localhost:5500",

    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "model": "linear_svm_tuned"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(job: JobPosting):
    model = ml_models.get("linear_svm_tuned")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is currently unavailable"
        )

    try:
        # Convert request to pandas DataFrame
        input_data = pd.DataFrame([job.model_dump()])

        # The trained pipeline expects the dataframe that comes
        # out of preprocess_data.
        processed_data, _ = preprocess_data(input_data)

        # Get decision function score for LinearSVC
        decision_score = model.decision_function(processed_data)[0]

        # Apply custom threshold
        fraudulent_int = 1 if decision_score >= THRESHOLD else 0

        prediction_label = (
            "Potentially Fraudulent"
            if fraudulent_int == 1
            else "Likely Legitimate"
        )

        return PredictionResponse(
            prediction=prediction_label,
            fraudulent=fraudulent_int,
            decision_score=float(decision_score)
        )

    except Exception as e:
        logger.error(f"Error during prediction: {e}")

        raise HTTPException(
            status_code=400,
            detail="Failed to process input data for prediction"
        )