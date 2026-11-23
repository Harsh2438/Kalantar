"""
KALANTAR: Simple Delay Prediction Test
--------------------------------------
Loads the saved model and predicts arrival delays for sample trains.
"""

import pickle
from pathlib import Path
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

def load_kalantar_model():
    """Loads the trained model and preprocessor using built-in pickle."""
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
        
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    return preprocessor, model

def predict_train_delay(journey_data: pd.DataFrame):
    """Encodes features and predicts delay in minutes."""
    preprocessor, model = load_kalantar_model()
    
    # 1. Convert text to numbers
    journey_encoded = preprocessor.transform(journey_data)
    
    # 2. Predict delay
    predicted_delays = model.predict(journey_encoded)
    return predicted_delays

def main():
    print("=" * 60)
    print("       KALANTAR RAILWAY INTELLIGENCE SYSTEM")
    print("           Live Delay Prediction Demo")
    print("=" * 60)
    
    # Scenario 1: Normal operational conditions (Punctual rake, clear weather)
    train_normal = {
        "train_type": "Superfast",
        "year": 2024,
        "month": 4,
        "day_of_week": 2,
        "departure_hour": 10,
        "is_weekend": 0,
        "is_night_departure": 0,
        "is_peak_hour": 1,
        "is_festival_season": 0,
        "season": "Summer",
        "zone": "NR",
        "source_station_category": "A1",
        "destination_station_category": "A1",
        "distance_km": 750,
        "num_scheduled_stops": 12,
        "scheduled_travel_hours": 11.5,
        "track_doubled": 1,
        "is_hdn_route": 1,
        "traction_type": "Electric",
        "is_electrified": 1,
        "psr_count": 3,
        "is_circular_route": 0,
        "is_monsoon_season": 0,
        "is_fog_risk": 0,
        "fog_risk_score": 0.0,
        "zone_fog_index": 0.0,
        "zone_congestion_index": 0.45,
        "season_severity_score": 0.15,
        "loco_age_years": 4,
        "coach_age_years": 3,
        "has_lhb_coaches": 1,
        "is_rake_shared": 0,
        "maintenance_score": 92.0,
        "seat_utilisation_pct": 85.0,
        "is_overloaded": 0,
        "late_incoming_rake": 0,       # Incoming rake arrived ON TIME
        "is_special_train": 0,
        "route_historical_ontime_pct": 82.5,
    }

    # Scenario 2: High disruption conditions (Late incoming rake + Winter fog)
    train_disrupted = {
        "train_type": "Express",
        "year": 2024,
        "month": 12,
        "day_of_week": 5,
        "departure_hour": 23,
        "is_weekend": 0,
        "is_night_departure": 1,
        "is_peak_hour": 0,
        "is_festival_season": 0,
        "season": "Winter",
        "zone": "NCR",
        "source_station_category": "A1",
        "destination_station_category": "A",
        "distance_km": 1100,
        "num_scheduled_stops": 22,
        "scheduled_travel_hours": 19.0,
        "track_doubled": 1,
        "is_hdn_route": 1,
        "traction_type": "Electric",
        "is_electrified": 1,
        "psr_count": 8,
        "is_circular_route": 0,
        "is_monsoon_season": 0,
        "is_fog_risk": 1,              # Winter fog present
        "fog_risk_score": 0.85,
        "zone_fog_index": 0.90,
        "zone_congestion_index": 0.88,
        "season_severity_score": 0.78,
        "loco_age_years": 11,
        "coach_age_years": 10,
        "has_lhb_coaches": 0,
        "is_rake_shared": 1,
        "maintenance_score": 68.0,
        "seat_utilisation_pct": 98.0,
        "is_overloaded": 1,
        "late_incoming_rake": 1,       # Incoming rake arrived LATE
        "is_special_train": 0,
        "route_historical_ontime_pct": 48.0,
    }

    test_df = pd.DataFrame([train_normal, train_disrupted])
    predictions = predict_train_delay(test_df)

    # Display results
    print("\n--- SCENARIO 1: Normal Conditions (Superfast, On-Time Rake, Clear Weather) ---")
    print(f"Predicted Arrival Delay: {predictions[0]:.1f} minutes")
    print("Status: Minor / Routine Delay (Easily absorbed by scheduled track buffers)")

    print("\n--- SCENARIO 2: Disrupted Conditions (Express, Late Rake + Heavy Fog) ---")
    print(f"Predicted Arrival Delay: {predictions[1]:.1f} minutes")
    print(f"Difference: +{predictions[1] - predictions[0]:.1f} extra minutes due to incoming rake delay and winter fog!")
    print("Status: High Risk of Missed Passenger Connections")
    print("=" * 60)

if __name__ == "__main__":
    main()
