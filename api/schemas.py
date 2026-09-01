from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

# ─── PREDICTION SCHEMAS ────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    """A single sensor tick from a vehicle (PMSM + Synthetic BMS Data)."""
    vehicle_id: str = Field(..., description="Unique identifier for the EV compute unit.")
    ambient: float = Field(..., description="Ambient temperature.")
    coolant: float = Field(..., description="Coolant temperature.")
    u_d: float = Field(..., description="Voltage d-component.")
    u_q: float = Field(..., description="Voltage q-component.")
    motor_speed: float = Field(..., description="Motor speed/RPM.")
    torque: float = Field(..., description="Motor torque.")
    i_d: float = Field(..., description="Current d-component.")
    i_q: float = Field(..., description="Current q-component.")
    pm: float = Field(..., description="Permanent Magnet temperature.")
    stator_yoke: float = Field(..., description="Stator yoke temperature.")
    stator_tooth: float = Field(..., description="Stator tooth temperature.")
    stator_winding: float = Field(..., description="Stator winding temperature.")
    profile_id: Optional[int] = Field(None, description="Driving profile ID.")
    battery_temp: Optional[float] = Field(None, description="Synthetic battery core temperature.")
    soc: Optional[float] = Field(None, ge=0, le=100, description="State of Charge (%).")
    power_draw: Optional[float] = Field(None, description="Total electrical power draw (W).")

class PredictResponse(BaseModel):
    """Prediction outcome for a sensor reading."""
    predicted_fault: int = Field(..., description="Predicted fault code (0-4).")
    fault_label: str = Field(..., description="Human-readable fault label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score.")
    probabilities: Dict[int, float] = Field(..., description="Probability distribution across all fault classes.")
    explanations: Dict[str, float] = Field(..., description="SHAP feature importance values for the prediction.")

class BatchPredictRequest(BaseModel):
    """Batch of sensor readings for scoring."""
    readings: List[SensorReading]

class BatchPredictResponse(BaseModel):
    """Results for a batch of readings."""
    predictions: List[PredictResponse]

# ─── FORECAST SCHEMAS ──────────────────────────────────────────────────────────

class ProjectedState(BaseModel):
    stator_winding_projected: float
    coolant_projected: float
    motor_speed_projected: float
    battery_temp_projected: Optional[float] = None

class ForecastResponse(BaseModel):
    """Pre-emptive fault forecast."""
    forecasted_fault: int
    confidence: float
    horizon_ticks: int
    projected_state: Optional[ProjectedState] = None
    current_prediction: Optional[int] = None
    warning: Optional[str] = None

# ─── VEHICLE SCHEMAS ───────────────────────────────────────────────────────────

class VehicleHealthResponse(BaseModel):
    vehicle_id: str
    health_score: float = Field(..., ge=0, le=100, description="100 is perfect health.")
    active_faults: List[int]
    last_reading_time: str

class FleetHealthResponse(BaseModel):
    total_vehicles: int
    healthy_vehicles: int
    at_risk_vehicles: int
    vehicles: List[VehicleHealthResponse]

# ─── MODEL METRICS SCHEMAS ─────────────────────────────────────────────────────

class ModelMetricsResponse(BaseModel):
    version: str
    metrics: Dict[str, Any]
