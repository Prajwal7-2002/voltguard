import random
from typing import Any, Dict

import pandas as pd

from config.settings import cfg

# Recommended-action keywords (see root_cause.analyzer) that command a load derate.
_DERATE_KEYWORDS = ("derate", "limit", "reduce", "cut", "throttle")


class VehicleSimulator:
    """
    Playback Engine for PMSM Real-World Telemetry.
    Reads sequentially from the Kaggle dataset to fake a live motor stream.
    Computes synthetic battery_temp and soc on each tick.
    """
    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.soc = 100.0  # Start with full charge
        self.derate_ticks_remaining = 0

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

        # An active derate scales the commanded load. Temperatures are still
        # replayed from the dataset, so this models the setpoint, not thermal response.
        speed_factor, current_factor = 1.0, 1.0
        if self.derate_ticks_remaining > 0:
            speed_factor = 1 - cfg.healing.throttle_rpm_reduction
            current_factor = 1 - cfg.healing.throttle_current_reduction

        # Compute synthetic battery temperature from power draw
        u_d = float(row.get("u_d", 0))
        u_q = float(row.get("u_q", 0))
        i_d = float(row.get("i_d", 0)) * current_factor
        i_q = float(row.get("i_q", 0)) * current_factor
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
            "motor_speed": float(row.get("motor_speed", 1000.0)) * speed_factor,
            "torque": float(row.get("torque", 0)) * current_factor,
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
            "derated": self.derate_ticks_remaining > 0,
        }
        return state

    def tick(self) -> Dict[str, Any]:
        """Advances the playback by 1 row mimicking real time."""
        state = self.get_state()
        if self.df is not None:
            self.current_idx = (self.current_idx + 1) % len(self.df)
        if self.derate_ticks_remaining > 0:
            self.derate_ticks_remaining -= 1
        return state

    def apply_effect(self, action: str) -> bool:
        """Apply a recommended action to the playback.

        Load-reducing actions derate motor speed, torque and current for
        ``cfg.healing.derate_ticks`` ticks. Returns True if a derate was applied.
        """
        if any(k in action.lower() for k in _DERATE_KEYWORDS):
            self.derate_ticks_remaining = cfg.healing.derate_ticks
            return True
        return False
