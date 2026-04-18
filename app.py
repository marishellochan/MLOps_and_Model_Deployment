from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
import joblib
import time
import os
import json
import uuid
import pandas as pd
import numpy as np


# global paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
METADATA_PATH = os.getenv("MODEL_METADATA_PATH", "models/model_metadata.json")
MODEL_METRICS: dict = {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0}
LOOKUP_PATH = os.getenv("LOOKUP_PATH", "data/raw/taxi_lookup.csv")

#features
def features(df: pd.DataFrame, taxi_lookup: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # time-based features
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["is_weekend"] = df["pickup_day_of_week"].isin([5, 6]).astype(int)

    # duration in minutes
    df["trip_duration_minutes"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # speed
    df["trip_speed_mph"] = np.where(
        df["trip_duration_minutes"] > 0,
        df["trip_distance"] / (df["trip_duration_minutes"] / 60),
        0
    )

    df["log_trip_distance"] = np.log1p(df["trip_distance"])
    df["fare_per_mile"] = np.where(
        df["trip_distance"] > 0,
        df["fare_amount"] / df["trip_distance"],
        0
    )

    df["fare_per_minute"] = np.where(
        df["trip_duration_minutes"] > 0,
        df["fare_amount"] / df["trip_duration_minutes"],
        0
    )

    # Borough lookups
    pickup_lookup = taxi_lookup[["LocationID", "Borough"]].rename(
        columns={"LocationID": "PULocationID", "Borough": "pickup_borough"}
    )
    dropoff_lookup = taxi_lookup[["LocationID", "Borough"]].rename(
        columns={"LocationID": "DOLocationID", "Borough": "dropoff_borough"}
    )

    df = df.merge(pickup_lookup, on="PULocationID", how="left")
    df = df.merge(dropoff_lookup, on="DOLocationID", how="left")

    df["pickup_borough"] = df["pickup_borough"].fillna("Unknown")
    df["dropoff_borough"] = df["dropoff_borough"].fillna("Unknown")

    return df

class TripInput(BaseModel):
    pickup_hour: int = Field(..., ge=0, le=23)
    pickup_day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: bool
    trip_duration_minutes: float = Field(..., gt=0)
    trip_speed_mph: float = Field(..., ge=0)
    log_trip_distance: float = Field(..., ge=0)
    passenger_count: int = Field(..., ge=1, le=6)
    trip_distance: float = Field(..., gt=0)
    fare_amount: float = Field(..., gt=0)
    PULocationID: int = Field(..., ge=1)
    DOLocationID: int = Field(..., ge=1)


class PredictionResponse(BaseModel):
    predicted_tip_amount: float
    model_version: str
    prediction_id: str

class BatchInput(BaseModel):
    records: list[TripInput] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    feature_names: list[str]
    training_metrics: dict


model = None
start_time = None
taxi_lookup_df = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, model_metadata, taxi_lookup_df, app_start_time

    ml_model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        model_metadata = json.load(f)

    taxi_lookup_df = pd.read_csv(LOOKUP_PATH)
    app_start_time = time.time()

    yield


app = FastAPI(
    title="Taxi Tip Prediction API",
    description="Predicts tip amount for NYC Yellow Taxi trips. COMP 3610 Assignment 4.",
    version="1.0.0",
    lifespan=lifespan,
)

def predict_one(record: TripInput) -> PredictionResponse:
    feature_names = model_metadata["feature_names"]

    row_dict = record.model_dump()
    df = pd.DataFrame([row_dict], columns=feature_names)

    pred = ml_model.predict(df)[0]

    return PredictionResponse(
        predicted_tip_amount=round(float(pred), 2),
        prediction_id=str(uuid.uuid4()),
        model_version=str(model_metadata["model_version"])
    )

@app.get("/")
def root():
    return {"message": "Taxi Tip Prediction API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TripInput):
   return predict_one(input_data)
@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    predictions = [predict_one(record) for record in batch.records]

    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        processing_time_ms=10
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "model_version": model_metadata.get("model_version"),
        "uptime_seconds": round(time.time() - start_time, 1) if start_time else 0
    }


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    metrics = model_metadata.get("training_metrics", {})

    return {
        "model_name": model_metadata.get("model_name"),
        "version": model_metadata.get("model_version"),
        "feature_names": model_metadata.get("feature_names", []),
        "training_metrics": {
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "r2": metrics.get("r2")
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later."
        }
    )