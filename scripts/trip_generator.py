"""Realistic Time-Series Trip Generator for VoltGuard.

Unlike random data generation, this script creates continuous driving "trips".
It simulates physical properties like thermal inertia, steady SOC drain,
and progressive fault development over time.

Usage:
    python data/trip_generator.py --trips 500 --output data/raw/train_trips.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import cfg


def generate_single_trip(trip_id: int, rng: np.random.RandomState, ticks: int = 100) -> list[dict]:
    """Generate a single sequential driving session."""

    # Randomly assign a fault outcome for this trip
    fault_type = rng.choice(
        ["none", "battery_over_temp", "motor_overcurrent", "low_soc"], p=[0.70, 0.10, 0.10, 0.10]
    )

    # ── Initial State ─────────────────────────────────────────────
    # Random realistic starting points
    temp = rng.uniform(25.0, 35.0)
    soc = rng.uniform(70.0, 95.0) if fault_type != "low_soc" else rng.uniform(25.0, 35.0)
    cycles = rng.randint(50, 400)

    # Fault triggering mechanism (faults happen mid-trip)
    fault_start_tick = (
        rng.randint(int(ticks * 0.3), int(ticks * 0.7)) if fault_type != "none" else -1
    )

    trip_data = []

    for tick in range(ticks):
        # ── Base Driving Dynamics ─────────────────────────────────
        # Simulate varying load (city driving with acceleration/braking)
        load_factor = np.sin(tick / 5.0) + rng.uniform(-0.5, 1.0)

        current = max(0, 10 + (load_factor * 15))  # Normal current 10-25A
        rpm = current * 200 + rng.uniform(-500, 500)  # RPM correlates to current

        # SOC drains slowly
        soc_drain_rate = 0.05 + (current * 0.001)
        soc = max(0.0, soc - soc_drain_rate)

        # Thermal inertia (temperature slowly rubber-bands towards target temp)
        target_temp = 35 + (current * 0.5)
        temp += (target_temp - temp) * 0.05  # 5% adjustment per tick

        fault_code = 0  # Default normal

        # ── Progressive Fault Injection ───────────────────────────
        if fault_type != "none" and tick >= fault_start_tick:
            # How deep into the fault are we? [0.0 to 1.0]
            severity = min(1.0, (tick - fault_start_tick) / 20.0)

            if fault_type == "battery_over_temp":
                # Temperature forcefully climbs over several ticks
                temp += 2.0 * severity
                if temp > cfg.battery.temp_high:
                    fault_code = 1

            elif fault_type == "motor_overcurrent":
                # Sudden massive spike in current and RPM
                if severity > 0.5:
                    current = rng.uniform(cfg.motor.current_overcurrent + 10, 100.0)
                    rpm = rng.uniform(9000, 12000)
                    fault_code = 2

            elif fault_type == "low_soc":
                # SOC artificially drops quickly
                soc = max(0.0, soc - (0.5 * severity))
                if soc < cfg.battery.soc_low:
                    fault_code = 1

        # Add some sensor noise
        rpm = max(0, rpm + rng.uniform(-50, 50))
        voltage = 35 + (soc / 100.0) * (50 - 35) + rng.uniform(-0.2, 0.2)

        # Append tick data
        trip_data.append(
            {
                "trip_id": trip_id,
                "tick": tick,
                "fault_code": fault_code,
                "battery_temp": temp,
                "current": current,
                "motor_rpm": rpm,
                "soc": soc,
                "voltage": voltage,
                "battery_level": round(soc, 0),
                "charging_cycles": cycles,
                # Metadata for analysis
                "injected_fault_type": fault_type,
            }
        )

    return trip_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic time-series EV trip data")
    parser.add_argument("--trips", type=int, default=500, help="Number of trips to simulate")
    parser.add_argument("--ticks", type=int, default=100, help="Ticks per trip (duration)")
    parser.add_argument(
        "--output", type=str, default="data/raw/train_trips.csv", help="Output path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Generating {args.trips} trips ({args.ticks} ticks each)...")

    rng = np.random.RandomState(args.seed)
    all_data = []

    for trip_id in range(args.trips):
        trip_data = generate_single_trip(trip_id, rng, args.ticks)
        all_data.extend(trip_data)
        if (trip_id + 1) % 100 == 0:
            print(f"  Generated {trip_id + 1}/{args.trips} trips")

    df = pd.DataFrame(all_data)

    # Save raw data
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nDone! Saved {len(df)} rows to {args.output}")
    print("Class distribution (Target: fault_code):")
    print(df["fault_code"].value_counts().to_string())


if __name__ == "__main__":
    main()
