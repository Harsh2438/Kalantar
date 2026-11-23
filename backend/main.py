"""
KALANTAR: Railway Intelligence & Decision Support API
----------------------------------------------------
FastAPI backend service serving:
1. Dynamic XGBoost train ETA delay predictions with uncertainty intervals
2. Network-level station delay propagation simulation
3. Passenger connection risk analysis and automated alternative train recommendations
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.simulator import (
    evaluate_connection_risk,
    find_alternative_trains,
    get_timetable,
    simulate_route_delays,
)

# Paths
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="KALANTAR Railway Intelligence API",
    description="Explainable & Probabilistic AI for Train Delay Prediction, Propagation, and Decision Support",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL = None
PREPROCESSOR = None
METADATA = {}


@app.on_event("startup")
def load_artifacts():
    global MODEL, PREPROCESSOR, METADATA
    try:
        with open(PREPROCESSOR_PATH, "rb") as f:
            PREPROCESSOR = pickle.load(f)
        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            METADATA = json.load(f)
        print("KALANTAR AI Model and Preprocessor loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load model artifacts: {e}")


# --------------------------------------------------------------------------
# Request & Response Schemas
# --------------------------------------------------------------------------
class JourneyInput(BaseModel):
    train_type: str = "Superfast"
    year: int = 2024
    month: int = 12
    day_of_week: int = 3
    departure_hour: int = 14
    is_weekend: int = 0
    is_night_departure: int = 0
    is_peak_hour: int = 1
    is_festival_season: int = 0
    season: str = "Winter"
    zone: str = "NR"
    source_station_category: str = "A1"
    destination_station_category: str = "A1"
    distance_km: float = 750.0
    num_scheduled_stops: int = 14
    scheduled_travel_hours: float = 12.0
    track_doubled: int = 1
    is_hdn_route: int = 1
    traction_type: str = "Electric"
    is_electrified: int = 1
    psr_count: int = 4
    is_circular_route: int = 0
    is_monsoon_season: int = 0
    is_fog_risk: int = 0
    fog_risk_score: float = 0.0
    zone_fog_index: float = 0.0
    zone_congestion_index: float = 0.50
    season_severity_score: float = 0.25
    loco_age_years: int = 5
    coach_age_years: int = 4
    has_lhb_coaches: int = 1
    is_rake_shared: int = 0
    maintenance_score: float = 85.0
    seat_utilisation_pct: float = 85.0
    is_overloaded: int = 0
    late_incoming_rake: int = 0
    is_special_train: int = 0
    route_historical_ontime_pct: float = 78.0


class RouteSimInput(BaseModel):
    train_no: str
    initial_delay_minutes: float = 60.0


class ConnectionRiskInput(BaseModel):
    inbound_train_no: str
    outbound_train_no: str
    transfer_station_code: str
    inbound_predicted_delay: float


# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "KALANTAR Railway Intelligence",
        "model_loaded": MODEL is not None,
        "validation_mae_min": METADATA.get("validation_mae_min", 34.65),
        "test_mae_min": METADATA.get("test_mae_min", 34.92),
    }


@app.get("/api/sample-trains")
def get_sample_trains():
    """Returns sample major trains for quick demo selection."""
    return [
        {"train_no": "12626", "name": "NDLS-TVC KERALA EXPRESS", "route": "New Delhi -> Trivandrum"},
        {"train_no": "12951", "name": "MUMBAI RAJDHANI", "route": "Mumbai Central -> New Delhi"},
        {"train_no": "12301", "name": "HOWRAH RAJDHANI", "route": "Howrah -> New Delhi"},
        {"train_no": "12183", "name": "BHOPAL EXPRESS", "route": "Bhopal -> Partapgarh"},
        {"train_no": "11078", "name": "JHELUM EXPRESS", "route": "Jammu Tawi -> Pune"},
    ]


@app.get("/api/popular-junctions")
def get_popular_junctions():
    """Returns popular railway junction stations for transfer demo."""
    return [
        {"code": "BPL", "name": "Bhopal Junction"},
        {"code": "NDLS", "name": "New Delhi"},
        {"code": "CNB", "name": "Kanpur Central"},
        {"code": "PUNE", "name": "Pune Junction"},
        {"code": "JHS", "name": "Jhansi Junction"},
        {"code": "AGC", "name": "Agra Cantt"},
        {"code": "MTJ", "name": "Mathura Junction"},
        {"code": "BINA", "name": "Bina Junction"},
    ]


@app.post("/api/predict-delay")
def predict_delay(journey: JourneyInput):
    """Predicts train arrival delay with empirical 80% uncertainty bounds."""
    if MODEL is None or PREPROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    data_dict = journey.dict()
    df_input = pd.DataFrame([data_dict])

    try:
        encoded = PREPROCESSOR.transform(df_input)
        raw_pred = float(MODEL.predict(encoded)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    # Uncertainty bounds from metadata
    offsets = METADATA.get("uncertainty_interval_80pct", {"lower_offset_min": -53.4, "upper_offset_min": 58.8})
    lower_bound = max(0.0, raw_pred + offsets["lower_offset_min"])
    upper_bound = max(0.0, raw_pred + offsets["upper_offset_min"])

    # Classification
    if raw_pred <= 15:
        category = "Punctual / Minor Delay"
        color = "green"
    elif raw_pred <= 60:
        category = "Moderate Delay"
        color = "amber"
    else:
        category = "Severe Delay"
        color = "red"

    # Identify primary contributing factors
    factors = []
    if journey.late_incoming_rake == 1:
        factors.append({"factor": "Late Incoming Rake", "impact": "High (+60 to +90 min typical)", "level": "critical"})
    if journey.is_fog_risk == 1 or journey.fog_risk_score > 0.5:
        factors.append({"factor": "Dense Fog Conditions", "impact": "High (+40 to +70 min typical)", "level": "critical"})
    if journey.route_historical_ontime_pct < 60:
        factors.append({"factor": "Historical Route Congestion", "impact": "Moderate (+25 min typical)", "level": "warning"})
    if journey.has_lhb_coaches == 0:
        factors.append({"factor": "Conventional ICF Rolling Stock", "impact": "Moderate (+15 min typical)", "level": "info"})
    if not factors:
        factors.append({"factor": "Standard Optimal Operations", "impact": "Buffer absorption in effect", "level": "safe"})

    return {
        "predicted_delay_min": round(raw_pred, 1),
        "uncertainty_range": {
            "lower_min": round(lower_bound, 1),
            "upper_min": round(upper_bound, 1),
            "confidence": "80% empirical interval",
        },
        "status_category": category,
        "status_color": color,
        "key_factors": factors,
    }


@app.post("/api/simulate-route")
def simulate_route(req: RouteSimInput):
    """Simulates how an initial delay propagates down a train's stops."""
    try:
        route_df = simulate_route_delays(req.train_no, req.initial_delay_minutes)
        return {
            "train_no": req.train_no,
            "initial_delay_min": req.initial_delay_minutes,
            "total_stops": len(route_df),
            "stops": route_df.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/evaluate-connection")
def connection_risk_endpoint(req: ConnectionRiskInput):
    """Evaluates transfer risk and finds alternative trains if connection is missed."""
    try:
        risk_report = evaluate_connection_risk(
            inbound_train_no=req.inbound_train_no,
            outbound_train_no=req.outbound_train_no,
            transfer_station_code=req.transfer_station_code,
            inbound_predicted_delay=req.inbound_predicted_delay,
        )

        alternatives = []
        if "MISSED" in risk_report["risk_level"] or "TIGHT" in risk_report["risk_level"]:
            alt_df = find_alternative_trains(
                transfer_station_code=req.transfer_station_code,
                missed_train_no=req.outbound_train_no,
                after_time_str=risk_report["estimated_arrival"],
                max_results=3,
            )
            if not alt_df.empty:
                alternatives = alt_df.to_dict(orient="records")

        return {
            "risk_report": risk_report,
            "alternatives": alternatives,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------------------------------------------------------------------------
# Serve Frontend Static Files
# --------------------------------------------------------------------------
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8050, reload=True)
