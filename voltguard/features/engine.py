"""Feature engineering and label generation for diagnostics models.

Leak-proof intent:
- Model inputs use upstream control/electrical signals.
- Ground-truth fault labels are generated from separate thermal channels
  when available to reduce direct label leakage from model inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Leak-proof upstream input features.
FEATURE_COLUMNS = [
    "ambient",
    "coolant",
    "u_d",
    "u_q",
    "motor_speed",
    "torque",
    "i_d",
    "i_q",
]

FAULT_LABELS = {
    0: "Nominal",
    1: "Stator Overheat",
    2: "Battery Thermal Stress",
    3: "Coolant System Failure",
    4: "Inverter Over-Current",
}


def _safe_group_diff(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute first-order diff per vehicle when vehicle_id exists."""
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    if "vehicle_id" in df.columns:
        return df.groupby("vehicle_id", sort=False)[col].diff().fillna(0)
    return df[col].diff().fillna(0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from raw telemetry."""
    df = df.copy()

    if "coolant" in df.columns and "ambient" in df.columns:
        df["thermal_delta"] = df["coolant"] - df["ambient"]

    if "coolant" in df.columns:
        df["dt_dt"] = _safe_group_diff(df, "coolant")

    if "motor_speed" in df.columns:
        df["acceleration_gradient"] = _safe_group_diff(df, "motor_speed")

    if all(c in df.columns for c in ["u_d", "i_d", "u_q", "i_q", "ambient"]):
        df["power_draw"] = (df["u_d"] * df["i_d"]) + (df["u_q"] * df["i_q"])
        power_contribution = (df["power_draw"].abs() * 0.002).clip(upper=40.0)
        df["battery_temp"] = df["ambient"] + power_contribution

    return df


ENGINEERED_COLUMNS = [
    "thermal_delta",
    "dt_dt",
    "acceleration_gradient",
    "power_draw",
    "battery_temp",
]

ALL_FEATURES = FEATURE_COLUMNS + ENGINEERED_COLUMNS


def generate_fault_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Generate 5-class labels with leakage-safe preference.

    Primary strategy:
    - Uses hidden thermal channels not in model inputs:
      stator_winding, stator_tooth, stator_yoke, pm.
    - Assigns proxy fault classes using robust quantile thresholds.

    Fallback strategy:
    - If hidden channels are unavailable, uses legacy rules.
    """
    df = df.copy()
    df["fault_code"] = 0

    hidden_channels = {"stator_winding", "stator_tooth", "stator_yoke", "pm"}
    has_hidden = hidden_channels.issubset(set(df.columns))

    if has_hidden:
        winding_q94 = df["stator_winding"].quantile(0.94)
        tooth_q96 = df["stator_tooth"].quantile(0.96)
        yoke_q95 = df["stator_yoke"].quantile(0.95)
        pm_q95 = df["pm"].quantile(0.95)
        pm_q97 = df["pm"].quantile(0.97)

        winding_hot = df["stator_winding"] >= winding_q94
        yoke_hot = df["stator_yoke"] >= yoke_q95
        tooth_hot = df["stator_tooth"] >= tooth_q96
        pm_hot = df["pm"] >= pm_q95
        pm_spike = df["pm"] >= pm_q97

        # Low priority first, high priority last.
        df.loc[pm_spike, "fault_code"] = 4
        df.loc[yoke_hot & pm_hot, "fault_code"] = 2
        df.loc[tooth_hot & yoke_hot, "fault_code"] = 3
        df.loc[winding_hot, "fault_code"] = 1
        return df

    # Fallback: legacy rules (less leak-safe).
    if "i_q" in df.columns and "i_d" in df.columns:
        iq_threshold = df["i_q"].mean() + (3.0 * df["i_q"].std())
        id_threshold = df["i_d"].mean() + (3.0 * df["i_d"].std())
        overcurrent = (df["i_q"].abs() > iq_threshold) | (df["i_d"].abs() > id_threshold)
        df.loc[overcurrent, "fault_code"] = 4

    if "battery_temp" in df.columns:
        df.loc[df["battery_temp"] > 60.0, "fault_code"] = 2

    if "coolant" in df.columns and "motor_speed" in df.columns:
        coolant_fail = (df["coolant"] > 70.0) & (df["motor_speed"].abs() < 500)
        df.loc[coolant_fail, "fault_code"] = 3

    if "stator_winding" in df.columns:
        threshold = df["stator_winding"].mean() + (1.5 * df["stator_winding"].std())
        df.loc[df["stator_winding"] > threshold, "fault_code"] = 1

    return df
