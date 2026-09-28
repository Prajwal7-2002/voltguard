"""Dataset summary for the dashboard (class balance, per-drive fault rates, input ranges)."""

from fastapi import APIRouter, HTTPException

from voltguard.diagnostics.registry import ModelRegistry
from voltguard.features.engine import FAULT_LABELS, FEATURE_COLUMNS
from voltguard.simulator.dataset import feature_ranges, load_replay_data, profile_summary

router = APIRouter(prefix="/data", tags=["Data"])

# Typical PMSM operating ranges, used for sliders when the dataset isn't present.
_DEFAULT_RANGES = {
    "ambient": (15.0, 25.0, 30.0),
    "coolant": (18.0, 30.0, 90.0),
    "u_d": (-130.0, -10.0, 60.0),
    "u_q": (-5.0, 60.0, 130.0),
    "motor_speed": (0.0, 2000.0, 6000.0),
    "torque": (-150.0, 10.0, 200.0),
    "i_d": (-240.0, -50.0, 0.0),
    "i_q": (-250.0, 10.0, 250.0),
}


@router.get("/summary")
def data_summary():
    """Class balance and drive coverage come from the served model's metrics; per-drive
    fault rates and input ranges come from the replay dataset when it is available."""
    registry = ModelRegistry()
    try:
        _, _, metrics, _ = registry.load("latest")
    except Exception as e:
        raise HTTPException(status_code=503, detail="No trained model available.") from e

    ranges = feature_ranges() or {
        col: {"min": lo, "median": mid, "max": hi} for col, (lo, mid, hi) in _DEFAULT_RANGES.items()
    }
    return {
        "dataset_available": load_replay_data() is not None,
        "fault_labels": {str(k): v for k, v in FAULT_LABELS.items()},
        "class_distribution": metrics.get("class_distribution", {}),
        "profiles_per_class": metrics.get("profiles_per_class", {}),
        "profiles": profile_summary() or [],
        "feature_ranges": {col: ranges[col] for col in FEATURE_COLUMNS},
    }
