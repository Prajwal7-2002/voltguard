from fastapi import APIRouter, HTTPException

from api.schemas import ModelMetricsResponse
from voltguard.diagnostics.registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["Models"])

registry = ModelRegistry()


@router.get("/latest", response_model=ModelMetricsResponse)
def get_latest_model():
    """Retrieve metrics for the latest trained model."""
    try:
        latest_version = registry.get_latest_version()
        _, _, metrics, _ = registry.load(latest_version)
        return ModelMetricsResponse(version=latest_version, metrics=metrics)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Latest model not found.") from e


@router.get("/{version}", response_model=ModelMetricsResponse)
def get_model_by_version(version: str):
    """Retrieve metrics for a specifically versioned model (e.g. v2)."""
    try:
        _, _, metrics, _ = registry.load(version)
        return ModelMetricsResponse(version=version, metrics=metrics)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model version {version} not found.") from e


@router.get("/{version}/drift")
def get_model_drift_status(version: str):
    """Check the current drift status of the specified model.
    In a real system, this would trigger an integration with the DriftDetector evaluating
    recent buffer tables directly to determine if an auto-retrain should fire.
    """
    from voltguard.diagnostics.drift import DriftDetector

    try:
        detector = DriftDetector(version=version)
        return {"version": version, "drift_capable": bool(detector.baseline)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Drift check failed.") from e
