import random
import pandas as pd
from typing import Dict, Any


class VehicleSimulator:
    """
    Playback Engine for PMSM Real-World Telemetry.
    Reads sequentially from the Kaggle dataset to fake a live motor stream.
    Computes synthetic battery_temp and soc on each tick.
    """
    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.soc = 100.0  # Start with full charge

        # Load the physical Kaggle dataset into memory for this vehicle
        try:
            self.df = pd.read_csv("data/comprehensive_fault_training_data.csv")
            # Start near a fault-prone region so the dashboard shows realistic variety
            # Find rows with high current or high coolant (fault-adjacent zones)
            high_stress = self.df[
                (self.df["i_q"].abs() > self.df["i_q"].quantile(0.85)) |
                (self.df["coolant"] > self.df["coolant"].quantile(0.85))
            ]
            if len(high_stress) > 100:
                # Start 50 rows before a high-stress region (to show transition)
                start = max(0, high_stress.index[random.randint(0, len(high_stress) - 1)] - 50)
                self.current_idx = start
            else:
                self.current_idx = random.randint(0, len(self.df) - 1000)
        except Exception:
            self.df = None
            self.current_idx = 0

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the exact physical state from the Kaggle array at the current tick.
        Adds synthetic battery_temp and soc computations.
        """
        if self.df is None:
            # Fallback if the dataset isn't loaded properly
            return {
                "vehicle_id": self.vehicle_id,
                "ambient": 35.0,
                "coolant": 40.0,
                "u_d": -0.5,
                "u_q": -0.5,
                "motor_speed": 1000.0,
                "torque": 15.0,
                "i_d": -1.0,
                "i_q": -1.0,
                "pm": 30.0,
                "stator_yoke": 30.0,
                "stator_tooth": 35.0,
                "stator_winding": 40.0,
                "profile_id": 4,
                "battery_temp": 38.0,
                "soc": self.soc,
            }

        # Get true physics from the dataset row
        row = self.df.iloc[self.current_idx]

        # Compute synthetic battery temperature from power draw
        u_d = float(row.get("u_d", 0))
        u_q = float(row.get("u_q", 0))
        i_d = float(row.get("i_d", 0))
        i_q = float(row.get("i_q", 0))
        ambient = float(row.get("ambient", 35.0))

        power_draw = (u_d * i_d) + (u_q * i_q)
        power_contribution = min(abs(power_draw) * 0.002, 40.0)
        battery_temp = ambient + power_contribution

        # Simulate SOC drain based on power consumption
        # Realistic: ~0.001% per tick at moderate load
        drain_rate = abs(power_draw) * 0.000005
        self.soc = max(0.0, self.soc - drain_rate)

        state = {
            "vehicle_id": self.vehicle_id,
            "ambient": ambient,
            "coolant": float(row.get("coolant", 40.0)),
            "u_d": u_d,
            "u_q": u_q,
            "motor_speed": float(row.get("motor_speed", 1000.0)),
            "torque": float(row.get("torque", 0)),
            "i_d": i_d,
            "i_q": i_q,
            "pm": float(row.get("pm", 30.0)),
            "stator_yoke": float(row.get("stator_yoke", 30.0)),
            "stator_tooth": float(row.get("stator_tooth", 35.0)),
            "stator_winding": float(row.get("stator_winding", 40.0)),
            "profile_id": int(row.get("profile_id", 4)),
            "battery_temp": round(battery_temp, 2),
            "soc": round(self.soc, 2),
            "power_draw": round(power_draw, 2),
        }
        return state

    def tick(self) -> Dict[str, Any]:
        """Advances the playback by 1 row mimicking real time."""
        state = self.get_state()
        if self.df is not None:
            self.current_idx = (self.current_idx + 1) % len(self.df)
        return state

    def apply_effect(self, action: str):
        """
        Intervenes in the playback to simulate self-healing.
        If we detect a fault, we can partially restore SOC or skip ahead.
        """
        # Simulate a partial recovery effect
        if "throttle" in action.lower() or "reduce" in action.lower():
            self.soc = min(100.0, self.soc + 0.5)  # Slight recovery from reduced load
