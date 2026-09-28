"""SQLAlchemy Database Schema Models."""

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from voltguard.database.config import Base


class VehicleDB(Base):
    """Corresponds to physical EV configurations and static info."""

    __tablename__ = "vehicles"

    id = Column(String, primary_key=True, index=True)
    oem_name = Column(String, nullable=True)
    health_score = Column(Float, default=100.0)
    battery_capacity_kwh = Column(Float, nullable=True)
    active_faults = Column(JSON, default=list)  # List of integer fault codes
    last_reading_time = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PredictionLog(Base):
    """Audit table logging every ML prediction for drift detection and legal compliance."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    vehicle_id = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    predicted_fault = Column(Integer)
    confidence = Column(Float)
    raw_payload = Column(JSON)  # Store incoming payload parameters natively
    root_cause_action = Column(String, nullable=True)
