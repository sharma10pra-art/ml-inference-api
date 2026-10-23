from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Real-Time ML Inference API",
    description="A simple Iris flower classifier API for the ML inference capstone.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    sepal_length: float = Field(gt=0, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, description="Sepal width in cm")
    petal_length: float = Field(gt=0, description="Petal length in cm")
    petal_width: float = Field(gt=0, description="Petal width in cm")

class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]

@app.get("/")
def home():
    return {
        "message": "ML Inference API is running",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    features = np.array([[
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ]], dtype=float)

    try:
        predicted_index = int(model.predict(features)[0])
        probabilities_array = model.predict_proba(features)[0]
        class_names = model.classes_
        probabilities = {
            str(class_names[i]): round(float(probabilities_array[i]), 6)
            for i in range(len(class_names))
        }
        return PredictionResponse(
            prediction=str(predicted_index),
            probabilities=probabilities,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
