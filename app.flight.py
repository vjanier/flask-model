"""Airline price prediction API.

Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
Needs airline_price_model.joblib and model_metadata.json in this folder
(or set MODEL_PATH / META_PATH env vars).
"""
import json
import os
from contextlib import asynccontextmanager
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = os.getenv("MODEL_PATH", "airline_price_model.joblib")
META_PATH = os.getenv("META_PATH", "model_metadata.json")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        state["meta"] = json.load(f)
    yield
    state.clear()


app = FastAPI(title="Airline Price Predictor", version="1.0", lifespan=lifespan)


class Flight(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    airline: str
    source_city: str
    departure_time: str
    stops: str = Field(description="zero | one | two_or_more")
    arrival_time: str
    destination_city: str
    flight_class: str = Field(alias="class", description="Economy | Business")
    duration: float = Field(gt=0, description="Flight duration in hours")
    days_left: int = Field(ge=0, description="Days until departure")


def validate(f: Flight) -> pd.DataFrame:
    meta = state["meta"]
    row = f.model_dump(by_alias=True)  # key 'class' matches training column
    errors = []
    for col, allowed in meta["categories"].items():
        if row.get(col) not in allowed:
            errors.append(f"{col}: '{row.get(col)}' not in {allowed}")
    lo, hi = meta["duration_range"]
    if not lo <= row["duration"] <= hi:
        errors.append(f"duration must be between {lo} and {hi}")
    lo, hi = meta["days_left_range"]
    if not lo <= row["days_left"] <= hi:
        errors.append(f"days_left must be between {lo} and {hi}")
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    return pd.DataFrame([row])[meta["feature_cols"]]


@app.get("/health")
def health():
    meta = state["meta"]
    return {
        "status": "ok",
        "trained_at": meta["trained_at"],
        "sklearn_version": meta["sklearn_version"],
        "test_scores": meta["test_scores"],
    }


@app.get("/options")
def options():
    """Valid category values and numeric ranges, for building a form."""
    meta = state["meta"]
    return {
        "categories": meta["categories"],
        "duration_range": meta["duration_range"],
        "days_left_range": meta["days_left_range"],
    }


@app.post("/predict")
def predict(f: Flight):
    df = validate(f)
    price = float(state["model"].predict(df)[0])
    return {"price": round(price, 2), "currency": state["meta"]["currency"]}


@app.post("/predict/batch")
def predict_batch(flights: List[Flight]):
    if not 0 < len(flights) <= 1000:
        raise HTTPException(422, "Send between 1 and 1000 flights")
    df = pd.concat([validate(f) for f in flights], ignore_index=True)
    prices = state["model"].predict(df)
    return {
        "currency": state["meta"]["currency"],
        "prices": [round(float(p), 2) for p in prices],
    }
