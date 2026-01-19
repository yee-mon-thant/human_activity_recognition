"""
FastAPI app for Human Activity Recognition
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="Human Activity Recognition API",
    description="Predict human activities from smartphone sensor data",
    version="1.0.0"
)

# Load model and features at startup
import os
model_path = os.getenv('MODEL_PATH', '../../notebook/models/model.pkl')
features_path = os.getenv('FEATURES_PATH', '../../notebook/models/features.pkl')

try:
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = None

# Activity mapping
ACTIVITIES = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SITTING',
    4: 'STANDING',
    5: 'LAYING'
}


# Request models
class PredictionRequest(BaseModel):
    features: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.5] * 561  # Example with 561 features
            }
        }


class BatchPredictionRequest(BaseModel):
    samples: List[List[float]]


# Response models
class PredictionResponse(BaseModel):
    activity: str
    activity_id: int
    confidence: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Human Activity Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/predict/batch": "POST - Batch predictions",
            "/health": "GET - Health check",
            "/activities": "GET - List all activities"
        }
    }


@app.get("/health")
def health_check():
    """Check if model is loaded and ready"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model": "loaded",
        "features": len(feature_names) if feature_names else 0
    }


@app.get("/activities")
def get_activities():
    """Get list of all possible activities"""
    return {
        "activities": ACTIVITIES,
        "total": len(ACTIVITIES)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predict activity from sensor features

    Expects 561 features as input
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate input
    if len(request.features) != 561:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 561 features, got {len(request.features)}"
        )

    try:
        # Convert to numpy array
        features_array = np.array([request.features])

        # Make prediction
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        confidence = float(probabilities[prediction])

        return PredictionResponse(
            activity=ACTIVITIES[prediction],
            activity_id=int(prediction),
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):
    """
    Predict activities for multiple samples

    Each sample should have 561 features
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate all samples
    for i, sample in enumerate(request.samples):
        if len(sample) != 561:
            raise HTTPException(
                status_code=400,
                detail=f"Sample {i}: Expected 561 features, got {len(sample)}"
            )

    try:
        # Convert to numpy array
        features_array = np.array(request.samples)

        # Make predictions
        predictions = model.predict(features_array)
        probabilities = model.predict_proba(features_array)

        # Format results
        results = []
        for pred, probs in zip(predictions, probabilities):
            results.append(
                PredictionResponse(
                    activity=ACTIVITIES[pred],
                    activity_id=int(pred),
                    confidence=float(probs[pred])
                )
            )

        return BatchPredictionResponse(predictions=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
