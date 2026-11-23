"""
KALANTAR: Train, Evaluate, and Save Delay Prediction Model
----------------------------------------------------------
This script:
1. Loads sample train data (2018-2022), validation data (2023), and test data (2024).
2. Encodes categories (text columns) into numbers.
3. Trains an XGBoost model to predict train delays (in minutes).
4. Tests accuracy on brand new 2024 journeys.
5. Saves the trained model using Python's standard 'pickle' (.pkl) format.
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Folder paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = BASE_DIR / "data" / "raw" / "ir_train.csv"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 38 features used to predict train delays
FEATURES = [
    "train_type",
    "year",
    "month",
    "day_of_week",
    "departure_hour",
    "is_weekend",
    "is_night_departure",
    "is_peak_hour",
    "is_festival_season",
    "season",
    "zone",
    "source_station_category",
    "destination_station_category",
    "distance_km",
    "num_scheduled_stops",
    "scheduled_travel_hours",
    "track_doubled",
    "is_hdn_route",
    "traction_type",
    "is_electrified",
    "psr_count",
    "is_circular_route",
    "is_monsoon_season",
    "is_fog_risk",
    "fog_risk_score",
    "zone_fog_index",
    "zone_congestion_index",
    "season_severity_score",
    "loco_age_years",
    "coach_age_years",
    "has_lhb_coaches",
    "is_rake_shared",
    "maintenance_score",
    "seat_utilisation_pct",
    "is_overloaded",
    "late_incoming_rake",
    "is_special_train",
    "route_historical_ontime_pct",
]

USECOLS = ["departure_date", "delay_minutes"] + FEATURES


def load_sampled_data():
    """Reads the dataset in chunks to save memory and splits by year."""
    print("Reading dataset from ir_train.csv...")
    samples = {"train": [], "val": [], "test": []}

    for chunk in pd.read_csv(CSV_PATH, usecols=USECOLS, chunksize=100_000):
        years = pd.to_numeric(chunk["year"], errors="coerce")

        # 2018–2022: Training data
        train_chunk = chunk.loc[years <= 2022]
        if len(train_chunk):
            samples["train"].append(
                train_chunk.sample(n=min(5_000, len(train_chunk)), random_state=42)
            )

        # 2023: Validation data
        val_chunk = chunk.loc[years == 2023]
        if len(val_chunk):
            samples["val"].append(
                val_chunk.sample(n=min(2_000, len(val_chunk)), random_state=42)
            )

        # 2024: Final Test data
        test_chunk = chunk.loc[years == 2024]
        if len(test_chunk):
            samples["test"].append(
                test_chunk.sample(n=min(2_000, len(test_chunk)), random_state=42)
            )

    train_df = pd.concat(samples["train"], ignore_index=True)
    val_df = pd.concat(samples["val"], ignore_index=True)
    test_df = pd.concat(samples["test"], ignore_index=True)

    return train_df, val_df, test_df


def main():
    train_df, val_df, test_df = load_sampled_data()

    X_train = train_df[FEATURES]
    y_train = train_df["delay_minutes"].astype(float)

    X_val = val_df[FEATURES]
    y_val = val_df["delay_minutes"].astype(float)

    X_test = test_df[FEATURES]
    y_test = test_df["delay_minutes"].astype(float)

    # Separate text (categorical) columns from numeric columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in FEATURES if c not in categorical_cols]

    # Preprocessor: converts text categories into numbers
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        sparse_threshold=1.0,
    )

    print("Encoding features and training XGBoost model...")
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)

    # Train XGBoost Regressor
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        max_bin=128,
        n_jobs=2,
        random_state=42,
    )
    model.fit(X_train_enc, y_train)

    # Evaluate Validation (2023)
    val_preds = model.predict(X_val_enc)
    val_mae = mean_absolute_error(y_val, val_preds)
    train_median = y_train.median()
    val_base_mae = mean_absolute_error(y_val, np.full(len(y_val), train_median))

    # Evaluate Test (2024)
    test_preds = model.predict(X_test_enc)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_base_mae = mean_absolute_error(y_test, np.full(len(y_test), train_median))

    # Calculate uncertainty offsets (80% confidence interval)
    val_residuals = y_val - val_preds
    lower_offset = float(np.percentile(val_residuals, 10))
    upper_offset = float(np.percentile(val_residuals, 90))

    print(f"Validation MAE: {val_mae:.2f} min (Baseline: {val_base_mae:.2f} min)")
    print(f"Test MAE:       {test_mae:.2f} min (Baseline: {test_base_mae:.2f} min)")

    # Save using standard Python pickle (.pkl)
    print("\nSaving model as standard Python pickle (.pkl) files...")
    model_path = MODELS_DIR / "model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    metadata_path = MODELS_DIR / "metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)

    metadata = {
        "model_name": "KALANTAR_XGBoost_ETA_Model",
        "features": FEATURES,
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "training_median_delay_min": float(train_median),
        "validation_mae_min": round(float(val_mae), 2),
        "test_mae_min": round(float(test_mae), 2),
        "uncertainty_interval_80pct": {
            "lower_offset_min": round(lower_offset, 2),
            "upper_offset_min": round(upper_offset, 2),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Success! Saved:")
    print(" - models/model.pkl")
    print(" - models/preprocessor.pkl")
    print(" - models/metadata.json")


if __name__ == "__main__":
    main()
