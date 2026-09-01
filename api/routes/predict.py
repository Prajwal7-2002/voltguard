from fastapi import APIRouter, Depends, HTTPException

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ForecastResponse,
    PredictResponse,
    SensorReading,
)
from voltguard.diagnostics.predictor import FaultPredictor
from voltguard.forecaster.engine import FaultForecaster

router = APIRouter(prefix="/predict", tags=["Prediction"])

_predictor = None
_forecaster = None


def get_predictor() -> FaultPredictor:
    global _predictor
    if _predictor is None:
        try:
            _predictor = FaultPredictor(version="latest")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load FaultPredictor model: {str(e)}")
    return _predictor


def get_forecaster() -> FaultForecaster:
    global _forecaster
    if _forecaster is None:
        try:
            _forecaster = FaultForecaster(predictor_version="latest")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load Forecaster: {str(e)}")
    return _forecaster


@router.post("", response_model=PredictResponse)
def predict_single(reading: SensorReading, predictor: FaultPredictor = Depends(get_predictor)):
    """Predict fault code for a single vehicle sensor tick."""
    raw_data = reading.model_dump(exclude={"vehicle_id"}, exclude_none=True)
    try:
        result = predictor.predict(reading.vehicle_id, raw_data)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@router.post("/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest, predictor: FaultPredictor = Depends(get_predictor)):
    """Predict fault codes for a batch of readings sequentially."""
    results = []
    for reading in request.readings:
        raw_data = reading.model_dump(exclude={"vehicle_id"}, exclude_none=True)
        res = predictor.predict(reading.vehicle_id, raw_data)
        results.append(PredictResponse(**res))
    return BatchPredictResponse(predictions=results)


@router.post("/forecast", response_model=ForecastResponse)
def forecast_fault(reading: SensorReading, forecaster: FaultForecaster = Depends(get_forecaster)):
    """Forecast if a fault is likely in the near horizon."""
    raw_data = reading.model_dump(exclude={"vehicle_id"}, exclude_none=True)
    try:
        result = forecaster.forecast(reading.vehicle_id, raw_data)
        return ForecastResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Forecasting error: {str(e)}")
