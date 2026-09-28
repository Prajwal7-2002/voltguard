"""VoltGuard configuration — loads thresholds.yaml and provides typed access."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Resolve the path to thresholds.yaml relative to this file, so it works
# regardless of the working directory.
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_THRESHOLDS_PATH = _CONFIG_DIR / "thresholds.yaml"


# ---------------------------------------------------------------------------
# Pydantic models — typed, validated, auto-documented
# ---------------------------------------------------------------------------

class BatteryThresholds(BaseModel):
    soc_low: float = 25.0
    soc_critical: float = 10.0
    voltage_low: float = 36.0
    voltage_high: float = 50.5
    voltage_critical_low: float = 30.0
    temp_high: float = 60.0
    temp_low: float = 5.0
    temp_critical_high: float = 75.0
    temp_critical_low: float = -10.0
    charging_cycles_warn: int = 600
    charging_cycles_critical: int = 800


class MotorThresholds(BaseModel):
    current_overcurrent: float = 50.0
    current_critical: float = 80.0
    rpm_overspeed: float = 9000.0
    rpm_critical: float = 11000.0


class HealingParams(BaseModel):
    cooling_rate: float = 2.0
    cooling_duration: int = 5
    throttle_rpm_reduction: float = 0.3
    throttle_current_reduction: float = 0.4
    derate_ticks: int = 10
    recharge_soc_rate: float = 5.0
    preheat_rate: float = 1.5


class SimulationParams(BaseModel):
    tick_interval_seconds: float = 1.0
    default_num_vehicles: int = 5
    default_num_ticks: int = 100


class ModelParams(BaseModel):
    test_split: float = 0.2
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1


class VoltGuardConfig(BaseModel):
    """Top-level configuration container."""

    battery: BatteryThresholds = BatteryThresholds()
    motor: MotorThresholds = MotorThresholds()
    healing: HealingParams = HealingParams()
    simulation: SimulationParams = SimulationParams()
    model: ModelParams = ModelParams()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> VoltGuardConfig:
    """Load configuration from a YAML file.

    Falls back to defaults if the file is missing or the env var
    ``VOLTGUARD_CONFIG`` is not set.
    """
    if path is None:
        path = os.environ.get("VOLTGUARD_CONFIG", str(_DEFAULT_THRESHOLDS_PATH))

    path = Path(path)

    if path.exists():
        with open(path, "r") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return VoltGuardConfig(**raw)

    # No file found — use defaults
    return VoltGuardConfig()


# ---------------------------------------------------------------------------
# Module-level singleton so other modules can just ``from config.settings import cfg``
# ---------------------------------------------------------------------------
cfg: VoltGuardConfig = load_config()
