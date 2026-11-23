"""
KALANTAR: Railway Network Delay Propagation & Passenger Decision Engine
-----------------------------------------------------------------------
This module:
1. Traces a train's journey stop-by-stop from the official timetable.
2. Simulates how a delay propagates down the route (accounting for buffer recovery).
3. Evaluates passenger connection risks at junction stations.
4. Automatically recommends alternative trains if a connection is missed.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TIMETABLE_PATH = BASE_DIR / "data" / "processed" / "railway_timetable_clean.csv"

_TIMETABLE_CACHE = None

def get_timetable() -> pd.DataFrame:
    """Loads and caches the cleaned railway timetable."""
    global _TIMETABLE_CACHE
    if _TIMETABLE_CACHE is None:
        _TIMETABLE_CACHE = pd.read_csv(TIMETABLE_PATH)
        _TIMETABLE_CACHE["Train No"] = _TIMETABLE_CACHE["Train No"].astype(str).str.strip()
        _TIMETABLE_CACHE["Station Code"] = _TIMETABLE_CACHE["Station Code"].astype(str).str.strip()
    return _TIMETABLE_CACHE


def time_str_to_minutes(time_str: str) -> int:
    """Converts 'HH:MM:SS' string into minutes since midnight."""
    try:
        parts = [int(p) for p in str(time_str).split(":")]
        return parts[0] * 60 + parts[1]
    except Exception:
        return 0


def minutes_to_time_str(total_minutes: int) -> str:
    """Converts minutes since midnight back into 'HH:MM' format."""
    total_minutes = int(total_minutes) % 1440
    hours = total_minutes // 60
    mins = total_minutes % 60
    return f"{hours:02d}:{mins:02d}"


def simulate_route_delays(train_no: str, initial_delay_minutes: float) -> pd.DataFrame:
    """
    Simulates how an initial delay propagates down a train's scheduled route.
    Accounts for schedule buffer recovery (trains recover ~1.5 min per 100 km).
    """
    df = get_timetable()
    route = df[df["Train No"] == str(train_no)].sort_values("SEQ").copy()
    
    if route.empty:
        raise ValueError(f"Train {train_no} not found in the timetable.")
        
    current_delay = float(initial_delay_minutes)
    simulated_records = []
    prev_distance = route.iloc[0]["Distance"]
    
    for _, stop in route.iterrows():
        segment_km = max(0.0, float(stop["Distance"]) - float(prev_distance))
        prev_distance = stop["Distance"]
        
        # Buffer absorption: recover ~1.5 min per 100 km if running late
        buffer_recovery = (segment_km / 100.0) * 1.5
        current_delay = max(0.0, current_delay - buffer_recovery)
        
        arr_sched_min = time_str_to_minutes(stop["Arrival time"])
        arr_actual_min = arr_sched_min + current_delay
        
        simulated_records.append({
            "SEQ": int(stop["SEQ"]),
            "Station Code": stop["Station Code"],
            "Station Name": stop["Station Name"],
            "Scheduled Arrival": str(stop["Arrival time"])[:5],
            "Scheduled Departure": str(stop["Departure Time"])[:5],
            "Simulated Delay (min)": round(current_delay, 1),
            "Estimated Actual Arrival": minutes_to_time_str(arr_actual_min),
        })
        
    return pd.DataFrame(simulated_records)


def evaluate_connection_risk(
    inbound_train_no: str,
    outbound_train_no: str,
    transfer_station_code: str,
    inbound_predicted_delay: float
) -> dict:
    """
    Evaluates whether a passenger will safely make a train transfer at a junction station.
    """
    df = get_timetable()
    inbound = df[(df["Train No"] == str(inbound_train_no)) & (df["Station Code"] == str(transfer_station_code))]
    outbound = df[(df["Train No"] == str(outbound_train_no)) & (df["Station Code"] == str(transfer_station_code))]
    
    if inbound.empty:
        raise ValueError(f"Inbound train {inbound_train_no} does not stop at {transfer_station_code}.")
    if outbound.empty:
        raise ValueError(f"Outbound train {outbound_train_no} does not stop at {transfer_station_code}.")
        
    inbound_name = inbound.iloc[0]["Train Name"]
    outbound_name = outbound.iloc[0]["Train Name"]
    station_name = inbound.iloc[0]["Station Name"]
    
    inbound_arr_min = time_str_to_minutes(inbound.iloc[0]["Arrival time"])
    outbound_dep_min = time_str_to_minutes(outbound.iloc[0]["Departure Time"])
    
    # Calculate scheduled transfer buffer (handling midnight rollover)
    scheduled_buffer = outbound_dep_min - inbound_arr_min
    if scheduled_buffer < 0:
        scheduled_buffer += 1440
        
    # Effective transfer buffer after incoming delay
    effective_buffer = scheduled_buffer - inbound_predicted_delay
    
    # Risk classification
    if effective_buffer >= 25:
        risk_level = "LOW RISK (SAFE)"
        advice = "Comfortable transfer window. Proceed as scheduled."
    elif 10 <= effective_buffer < 25:
        risk_level = "MODERATE RISK (TIGHT)"
        advice = "Tight connection! Minimum platform transfer time. Walk quickly upon arrival."
    else:
        risk_level = "CRITICAL / MISSED CONNECTION"
        advice = "High probability of missed transfer. System will search for alternative trains."
        
    return {
        "transfer_station": f"{station_name} ({transfer_station_code})",
        "inbound_train": f"{inbound_train_no} ({inbound_name})",
        "outbound_train": f"{outbound_train_no} ({outbound_name})",
        "scheduled_arrival": str(inbound.iloc[0]["Arrival time"])[:5],
        "predicted_delay_min": round(inbound_predicted_delay, 1),
        "estimated_arrival": minutes_to_time_str(inbound_arr_min + inbound_predicted_delay),
        "scheduled_departure": str(outbound.iloc[0]["Departure Time"])[:5],
        "scheduled_buffer_min": scheduled_buffer,
        "effective_buffer_min": round(effective_buffer, 1),
        "risk_level": risk_level,
        "advice": advice,
    }


def find_alternative_trains(
    transfer_station_code: str,
    missed_train_no: str,
    after_time_str: str,
    max_results: int = 3
) -> pd.DataFrame:
    """
    Finds later trains departing from the transfer station that head to the same destination.
    Properly handles midnight rollover (e.g. departures within the next 12 hours).
    """
    df = get_timetable()
    missed_route = df[df["Train No"] == str(missed_train_no)].sort_values("SEQ")
    
    if missed_route.empty:
        return pd.DataFrame()
        
    dest_code = missed_route.iloc[-1]["Station Code"]
    dest_name = missed_route.iloc[-1]["Station Name"]
    after_min = time_str_to_minutes(after_time_str)
    
    # Get all trains stopping at the transfer station
    station_trains = df[df["Station Code"] == str(transfer_station_code)].copy()
    
    # Get all trains stopping at the destination station
    dest_trains = df[df["Station Code"] == str(dest_code)]["Train No"].unique()
    
    # Trains that serve BOTH stations
    shared_trains = set(station_trains["Train No"].unique()).intersection(set(dest_trains))
    shared_trains.discard(str(missed_train_no))
    
    candidates = []
    for t_no in shared_trains:
        t_route = df[df["Train No"] == t_no].sort_values("SEQ")
        t_dep_row = t_route[t_route["Station Code"] == str(transfer_station_code)].iloc[0]
        t_arr_row = t_route[t_route["Station Code"] == str(dest_code)].iloc[0]
        
        # Verify direction: departure must be before arrival
        if t_dep_row["SEQ"] < t_arr_row["SEQ"]:
            dep_time_str = str(t_dep_row["Departure Time"])[:5]
            dep_min = time_str_to_minutes(t_dep_row["Departure Time"])
            
            # Wait time accounting for midnight rollover
            wait_min = (dep_min - after_min) % 1440
            
            # Only suggest trains departing within the next 12 hours (720 min)
            if 0 < wait_min <= 720:
                candidates.append({
                    "Train No": t_no,
                    "Train Name": t_dep_row["Train Name"],
                    "Departure Time": dep_time_str,
                    "Destination": dest_name,
                    "Wait Time (hrs)": round(wait_min / 60.0, 1),
                })
                
    result_df = pd.DataFrame(candidates)
    if not result_df.empty:
        result_df = result_df.sort_values("Wait Time (hrs)").head(max_results)
        
    return result_df


def main():
    print("=" * 70)
    print("       KALANTAR DELAY PROPAGATION & DECISION SUPPORT SYSTEM")
    print("=" * 70)

    # Example Scenario:
    # Passenger arrives on Train 12626 (Kerala Express) at Bhopal Junction (BPL)
    # Connecting to Train 11078 (Jhelum Express to Pune)
    inbound_train = "12626"
    outbound_train = "11078"
    transfer_station = "BPL"

    print("\n--- 1. ROUTE DELAY PROPAGATION SIMULATION ---")
    initial_delay = 75.0  # Predicted by our XGBoost model
    print(f"Simulating Train {inbound_train} route with an initial {initial_delay} min delay:")
    route_sim = simulate_route_delays(inbound_train, initial_delay)
    print(route_sim.head(6).to_string(index=False))

    print("\n--- 2. PASSENGER CONNECTION RISK ASSESSMENT ---")
    risk_report = evaluate_connection_risk(
        inbound_train_no=inbound_train,
        outbound_train_no=outbound_train,
        transfer_station_code=transfer_station,
        inbound_predicted_delay=initial_delay
    )
    for k, v in risk_report.items():
        print(f"  * {k.replace('_', ' ').title()}: {v}")

    print("\n--- 3. PASSENGER DECISION SUPPORT & ALTERNATIVES ---")
    if "MISSED" in risk_report["risk_level"]:
        print(f"Searching for alternative trains from {risk_report['transfer_station']} to destination...")
        alts = find_alternative_trains(
            transfer_station_code=transfer_station,
            missed_train_no=outbound_train,
            after_time_str=risk_report["estimated_arrival"],
        )
        if not alts.empty:
            print("\nRecommended Alternative Trains:")
            print(alts.to_string(index=False))
        else:
            print("No direct alternatives found within the next 12 hours.")
    else:
        print("Transfer is safe. No alternative routing required.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
