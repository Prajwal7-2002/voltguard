from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ForecastResponse,
    PredictResponse,
    SensorReading,
)
from voltguard.core.logging import get_logger
from voltguard.database.config import get_db
from voltguard.database.models import PredictionLog, VehicleDB
from voltguard.diagnostics.predictor import FaultPredictor
from voltguard.forecaster.engine import FaultForecaster

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])

_predictor = None
_forecaster = None


def get_predictor() -> FaultPredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = FaultPredictor(version="latest")
        except Exception as e:
            logger.exception("Failed to load FaultPredictor")
            raise HTTPException(status_code=503, detail="Prediction model unavailable.") from e
    return _predictor


def get_forecaster() -> FaultForecaster:
    global _forecaster
    if _forecaster is None:
        try:
            _forecaster = FaultForecaster(predictor_version="latest")
        except Exception as e:
            logger.exception("Failed to load FaultForecaster")
            raise HTTPException(status_code=503, detail="Forecast model unavailable.") from e
    return _forecaster


def _record_prediction(db: Session, reading: SensorReading, result: dict) -> None:
    """Append to the audit log and refresh the vehicle's current health."""
    fault = result["predicted_fault"]
    db.add(
        PredictionLog(
            vehicle_id=reading.vehicle_id,
            predicted_fault=fault,
            confidence=result["confidence"],
            raw_payload=reading.model_dump(exclude_none=True),
        )
    )
    vehicle = db.get(VehicleDB, reading.vehicle_id)
    if vehicle is None:
        vehicle = VehicleDB(id=reading.vehicle_id)
        db.add(vehicle)
    vehicle.health_score = round(100 * float(result["probabilities"].get(0, 0.0)), 2)
    vehicle.active_faults = [fault] if fault != 0 else []


def _run_prediction(predictor: FaultPredictor, reading: SensorReading) -> dict:
    raw_data = reading.model_dump(exclude={"vehicle_id"}, exclude_none=True)
    return predictor.predict(reading.vehicle_id, raw_data)


@router.post("", response_model=PredictResponse)
def predict_single(
    reading: SensorReading,
    predictor: FaultPredictor = Depends(get_predictor),
    db: Session = Depends(get_db),
):
    """Predict fault code for a single vehicle sensor tick."""
    try:
        result = _run_prediction(predictor, reading)
    except Exception as e:
        logger.exception("Prediction failed for vehicle %s", reading.vehicle_id)
        raise HTTPException(status_code=500, detail="Prediction failed.") from e
    _record_prediction(db, reading, result)
    db.commit()
    return PredictResponse(**result)


@router.post("/batch", response_model=BatchPredictResponse)
def predict_batch(
    request: BatchPredictRequest,
    predictor: FaultPredictor = Depends(get_predictor),
    db: Session = Depends(get_db),
):
    """Predict fault codes for a batch of readings, in order, as one transaction."""
    results = []
    for i, reading in enumerate(request.readings):
        try:
            result = _run_prediction(predictor, reading)
        except Exception as e:
            db.rollback()
            logger.exception("Batch prediction failed at index %d", i)
            raise HTTPException(
                status_code=500, detail=f"Prediction failed for reading at index {i}."
            ) from e
        _record_prediction(db, reading, result)
        results.append(PredictResponse(**result))
    db.commit()
    return BatchPredictResponse(predictions=results)


@router.post("/forecast", response_model=ForecastResponse)
def forecast_fault(reading: SensorReading, forecaster: FaultForecaster = Depends(get_forecaster)):
    """Forecast if a fault is likely in the near horizon."""
    raw_data = reading.model_dump(exclude={"vehicle_id"}, exclude_none=True)
    try:
        result = forecaster.forecast(reading.vehicle_id, raw_data)
    except Exception as e:
        logger.exception("Forecast failed for vehicle %s", reading.vehicle_id)
        raise HTTPException(status_code=500, detail="Forecast failed.") from e
    return ForecastResponse(**result)
