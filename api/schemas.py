from typing import Any

from pydantic import BaseModel, Field

# ─── PREDICTION SCHEMAS ────────────────────────────────────────────────────────


class SensorReading(BaseModel):
    """A single sensor tick from a vehicle (PMSM + Synthetic BMS Data)."""

    vehicle_id: str = Field(
        ..., min_length=1, description="Unique identifier for the EV compute unit."
    )
    ambient: float = Field(..., ge=-80, le=100, description="Ambient temperature in C.")
    coolant: float = Field(..., ge=-80, le=180, description="Coolant temperature in C.")
    u_d: float = Field(..., ge=-1000, le=1000, description="Voltage d-component in V.")
    u_q: float = Field(..., ge=-1000, le=1000, description="Voltage q-component in V.")
    motor_speed: float = Field(..., description="Motor speed/RPM.")
    torque: float = Field(..., description="Motor torque.")
    i_d: float = Field(..., ge=-1000, le=1000, description="Current d-component in A.")
    i_q: float = Field(..., ge=-1000, le=1000, description="Current q-component in A.")
    pm: float = Field(..., ge=-80, le=250, description="Permanent Magnet temperature in C.")
    stator_yoke: float = Field(..., ge=-80, le=250, description="Stator yoke temperature in C.")
    stator_tooth: float = Field(..., ge=-80, le=250, description="Stator tooth temperature in C.")
    stator_winding: float = Field(
        ..., ge=-80, le=250, description="Stator winding temperature in C."
    )
    profile_id: int | None = Field(None, description="Driving profile ID.")
    battery_temp: float | None = Field(None, description="Synthetic battery core temperature.")
    soc: float | None = Field(None, ge=0, le=100, description="State of Charge (%).")
    power_draw: float | None = Field(None, description="Total electrical power draw (W).")


class PredictResponse(BaseModel):
    """Prediction outcome for a sensor reading."""

    predicted_fault: int = Field(..., description="Predicted fault code (0-4).")
    fault_label: str = Field(..., description="Human-readable fault label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score.")
    probabilities: dict[int, float] = Field(
        ..., description="Probability distribution across all fault classes."
    )
    explanations: dict[str, float] = Field(
        ..., description="SHAP feature importance values for the prediction."
    )


class BatchPredictRequest(BaseModel):
    """Batch of sensor readings for scoring."""

    readings: list[SensorReading] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    """Results for a batch of readings."""

    predictions: list[PredictResponse]


# ─── FORECAST SCHEMAS ──────────────────────────────────────────────────────────


class ProjectedState(BaseModel):
    stator_winding_projected: float
    coolant_projected: float
    motor_speed_projected: float
    battery_temp_projected: float | None = None


class ForecastResponse(BaseModel):
    """Pre-emptive fault forecast."""

    forecasted_fault: int
    confidence: float
    horizon_ticks: int
    projected_state: ProjectedState | None = None
    current_prediction: int | None = None
    warning: str | None = None


# ─── VEHICLE SCHEMAS ───────────────────────────────────────────────────────────


class VehicleHealthResponse(BaseModel):
    vehicle_id: str
    health_score: float = Field(..., ge=0, le=100, description="100 is perfect health.")
    active_faults: list[int]
    last_reading_time: str


class FleetHealthResponse(BaseModel):
    total_vehicles: int
    healthy_vehicles: int
    at_risk_vehicles: int
    vehicles: list[VehicleHealthResponse]


# ─── MODEL METRICS SCHEMAS ─────────────────────────────────────────────────────


class ModelMetricsResponse(BaseModel):
    version: str
    metrics: dict[str, Any]


# ─── EXPLAINABILITY SCHEMAS ────────────────────────────────────────────────────


class ExplainRequest(BaseModel):
    """A single what-if scenario over the raw model inputs (no history is kept)."""

    ambient: float = Field(..., ge=-80, le=100)
    coolant: float = Field(..., ge=-80, le=180)
    u_d: float = Field(..., ge=-1000, le=1000)
    u_q: float = Field(..., ge=-1000, le=1000)
    motor_speed: float = Field(..., ge=-20000, le=20000)
    torque: float = Field(..., ge=-1000, le=1000)
    i_d: float = Field(..., ge=-1000, le=1000)
    i_q: float = Field(..., ge=-1000, le=1000)


class ExplainResponse(PredictResponse):
    root_cause: str | None = None
    action: str | None = None
