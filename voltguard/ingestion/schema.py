"""Pydantic schemas for sensor data validation.

Every data point entering VoltGuard — whether from the simulator, a CSV
upload, or a REST API call — is validated through these schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SensorReading(BaseModel):
    """A single CAN bus sensor reading from an EV 2-wheeler."""

    battery_temp: float = Field(..., description="Battery temperature in °C")
    current: float = Field(..., description="Current in Amps (negative=discharge, positive=charge)")
    motor_rpm: float = Field(..., ge=0, description="Motor RPM")
    soc: float = Field(..., ge=0, le=100, description="State of charge (%)")
    voltage: float = Field(..., gt=0, description="Battery pack voltage (V)")
    battery_level: float = Field(..., description="Reported battery level (%)")
    charging_cycles: int = Field(..., ge=0, description="Total charge/discharge cycles")

    @field_validator("battery_temp")
    @classmethod
    def validate_temp_range(cls, v: float) -> float:
        if not -30 <= v <= 100:
            raise ValueError(f"battery_temp {v}°C outside physical range [-30, 100]")
        return v

    @field_validator("voltage")
    @classmethod
    def validate_voltage_range(cls, v: float) -> float:
        if not 20 <= v <= 60:
            raise ValueError(f"voltage {v}V outside expected range [20, 60]")
        return v


class PredictionResult(BaseModel):
    """Output schema for a fault prediction."""

    predicted_fault: int = Field(..., ge=0, le=2, description="0=No fault, 1=Battery, 2=Motor")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    explanations: dict[str, float] = Field(
        ..., description="SHAP values per feature for the predicted class"
    )
    input: dict[str, float] = Field(..., description="Original input data")


class HealingAction(BaseModel):
    """Output schema for a self-healing action."""

    fault_code: int
    root_cause: str
    action: str
    vehicle_id: str = ""
    tick: int = 0


class VehicleState(BaseModel):
    """Snapshot of a vehicle's current state (used by simulator)."""

    vehicle_id: str
    battery_temp: float = 30.0
    current: float = 0.0
    motor_rpm: float = 0.0
    soc: float = 85.0
    voltage: float = 48.0
    battery_level: float = 85.0
    charging_cycles: int = 200
    is_cooling: bool = False
    cooling_ticks_remaining: int = 0
    is_throttled: bool = False
    is_preheating: bool = False
    preheat_ticks_remaining: int = 0

    def to_sensor_reading(self) -> SensorReading:
        """Convert vehicle state to a sensor reading for prediction."""
        return SensorReading(
            battery_temp=self.battery_temp,
            current=self.current,
            motor_rpm=self.motor_rpm,
            soc=self.soc,
            voltage=self.voltage,
            battery_level=self.battery_level,
            charging_cycles=self.charging_cycles,
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to flat dict for the prediction pipeline."""
        return {
            "battery_temp": self.battery_temp,
            "current": self.current,
            "motor_rpm": self.motor_rpm,
            "soc": self.soc,
            "voltage": self.voltage,
            "battery_level": self.battery_level,
            "charging_cycles": self.charging_cycles,
        }
