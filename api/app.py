"""
FastAPI application serving the Day-Ahead Electricity Load Forecaster model.
Defines endpoints for point/interval predictions, batch inference, model info, retraining, and health diagnostics.
"""

import os
import pickle
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="Day-Ahead Electricity Load Forecaster API",
    description="Production-grade API serving LightGBM load forecasting models with 90% confidence intervals.",
    version="1.0.0"
)

# Relative Paths inside container / project
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "models", "forecaster_lgb.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "config.json")

# Global Forecaster reference
forecaster = None

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'DayAheadLoadForecaster':
            from src.models import DayAheadLoadForecaster
            return DayAheadLoadForecaster
        return super().find_class(module, name)

def load_model():
    global forecaster
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                forecaster = CustomUnpickler(f).load()
            logger.info("Forecaster model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load serialized model: {str(e)}")
            forecaster = None
    else:
        logger.warning(f"Model file not found at {MODEL_PATH}. Retraining may be required.")
        forecaster = None

# Load model on startup
@app.on_event("startup")
def startup_event():
    load_model()


# Pydantic Schemas for Predict requests
class HourlyReading(BaseModel):
    timestamp: str = Field(..., description="ISO 8601 string or YYYY-MM-DD HH:MM:SS")
    Global_active_power: float
    Global_reactive_power: float
    Voltage: float
    Global_intensity: float
    Sub_metering_1: float
    Sub_metering_2: float
    Sub_metering_3: float

class PredictRequest(BaseModel):
    history: List[HourlyReading] = Field(..., description="Strictly requires the previous 168 hours (7 days) of hourly readings to construct rolling/lag features.")

class MeterBatchRequest(BaseModel):
    meter_id: str
    history: List[HourlyReading]

class BatchPredictRequest(BaseModel):
    meters: List[MeterBatchRequest]


# Helper to construct features from raw history list
def build_features_from_history(history_list: List[HourlyReading]) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    # 1. Convert to DataFrame
    df = pd.DataFrame([r.dict() for r in history_list])
    df['Datetime'] = pd.to_datetime(df['timestamp'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    
    if len(df) < 168:
        raise HTTPException(
            status_code=400,
            detail=f"Incomplete history history. Model requires exactly 168 hours of history. Got {len(df)} hours."
        )
        
    # We only need the last row (forecast origin t) to make the prediction,
    # but we need the preceding 168 rows to construct the rolling means and lags.
    # Let's import create_features_and_targets from src.features
    from src.features import create_features_and_targets
    
    # Preprocess short-gap imputation flags if not present
    for col in ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
        df[f"{col}_is_short_imputed"] = 0
        df[f"{col}_is_profile_imputed"] = 0
        
    X, _ = create_features_and_targets(df, outlier_clipping=False)
    
    # We take the very last row, which is our forecast origin t
    if len(X) == 0:
         raise HTTPException(
            status_code=400,
            detail="Feature engineering failed to construct valid features. Ensure raw inputs do not contain NaNs."
        )
         
    X_origin = X.tail(1)
    return X_origin, X_origin.index


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/health", tags=["Diagnostics"])
def health():
    """
    Diagnostics check returning API status and model availability.
    """
    model_loaded = (forecaster is not None)
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info", tags=["Metadata"])
def model_info():
    """
    Returns metadata about the active ML model, features, and config parameters.
    """
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(status_code=404, detail="Configuration metadata not found.")
        
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        
    return {
        "algorithm": "LightGBM Direct Multi-Step Regression",
        "horizons_predicted": 24,
        "hyperparameters": config.get("hyperparameters", {}),
        "num_features_trained": config.get("num_features", 0),
        "training_timestamp": config.get("timestamp"),
        "training_duration_seconds": config.get("training_time_seconds")
    }


@app.get("/metrics", tags=["Metadata"])
def get_metrics():
    """
    Returns the out-of-sample test evaluation metrics (MAE, RMSE, MAPE, R2).
    """
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Evaluation metrics file not found.")
        
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
        
    return metrics


def retrain_model_task():
    logger.info("Background task: Initiating model retraining...")
    from src.train import main as run_training
    try:
        run_training()
        load_model()
        logger.info("Background task: Model retrained and reloaded successfully.")
    except Exception as e:
        logger.error(f"Retraining task failed: {str(e)}")


@app.post("/train", tags=["Retraining"])
def train(background_tasks: BackgroundTasks):
    """
    Asynchronously triggers model retraining on the server as a background task.
    """
    background_tasks.add_task(retrain_model_task)
    return {
        "status": "retraining_initiated",
        "message": "Model training running in background. Check API logs for status updates."
    }


@app.post("/predict", tags=["Inference"])
def predict(request: PredictRequest):
    """
    Generates 24-hour forecasts (point estimates and 90% confidence intervals) 
    given the previous 168 hours of historical readings.
    """
    global forecaster
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecaster model is not loaded. Trigger training first.")
        
    # Extract features
    X_origin, origin_time = build_features_from_history(request.history)
    
    # Predict
    point_preds, lower_preds, upper_preds = forecaster.predict(X_origin)
    
    # Reconstruct timestamps for the 24 predicted horizons
    t_start = origin_time[0]
    forecast_timestamps = [(t_start + pd.Timedelta(hours=h)).isoformat() for h in range(1, 25)]
    
    # Map predictions
    forecast = []
    for h in range(24):
        forecast.append({
            "timestamp": forecast_timestamps[h],
            "horizon_h": h + 1,
            "point_pred_kw": float(point_preds[0, h]),
            "lower_bound_90pct_kw": float(lower_preds[0, h]),
            "upper_bound_90pct_kw": float(upper_preds[0, h])
        })
        
    return {
        "forecast_origin_timestamp": t_start.isoformat(),
        "predictions": forecast
    }


@app.post("/batch_predict", tags=["Inference"])
def batch_predict(request: BatchPredictRequest):
    """
    Generates point and interval forecasts for a batch of multiple smart meters.
    """
    global forecaster
    if forecaster is None:
        raise HTTPException(status_code=503, detail="Forecaster model is not loaded.")
        
    results = []
    
    for meter in request.meters:
        try:
            X_origin, origin_time = build_features_from_history(meter.history)
            point_preds, lower_preds, upper_preds = forecaster.predict(X_origin)
            
            t_start = origin_time[0]
            forecast = []
            for h in range(24):
                forecast.append({
                    "horizon_h": h + 1,
                    "point_pred_kw": float(point_preds[0, h]),
                    "lower_bound_90pct_kw": float(lower_preds[0, h]),
                    "upper_bound_90pct_kw": float(upper_preds[0, h])
                })
            results.append({
                "meter_id": meter.meter_id,
                "forecast_origin_timestamp": t_start.isoformat(),
                "predictions": forecast
            })
        except Exception as e:
            results.append({
                "meter_id": meter.meter_id,
                "status": "failed",
                "error": str(e)
            })
            
    return {"batch_results": results}
