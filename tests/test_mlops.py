import numpy as np
import pandas as pd

from voltguard.diagnostics.drift import DriftDetector
from voltguard.features.engine import ALL_FEATURES


def test_drift_detector_initialization():
    """Verify DriftDetector loads the baseline seamlessly."""
    detector = DriftDetector("latest")
    assert detector.baseline is not None, "Baseline missing. Did you train a model first?"
    # Check that baseline contains at least some of our features
    assert "ambient" in detector.baseline


def test_nominal_distribution():
    """Verify unchanged distributions return nominal drift scores."""
    detector = DriftDetector("latest")

    # Spoof exact copies of mean distribution from baseline
    mock_data = {}
    for col in ALL_FEATURES:
        if col in detector.baseline:
            mock_data[col] = np.random.normal(loc=detector.baseline[col]["mean"], scale=1)
        else:
            mock_data[col] = 0.0
    df_nominal = pd.DataFrame([mock_data for _ in range(50)])

    result = detector.calculate_drift_score(df_nominal)
    assert not result["threshold_exceeded"]
    assert result["status"] == "nominal"


def test_critical_drift_distribution():
    """Verify heavily skewed distribution natively triggers retrain alert."""
    detector = DriftDetector("latest")

    # Spoof massively shifted distribution (e.g. 5 standard deviations away)
    mock_data = {}
    for col in ALL_FEATURES:
        if col in detector.baseline:
            mock_data[col] = np.random.normal(
                loc=detector.baseline[col]["mean"] + (detector.baseline[col]["std"] * 5), scale=1
            )
        else:
            mock_data[col] = 100.0
    df_drifted = pd.DataFrame([mock_data for _ in range(50)])

    result = detector.calculate_drift_score(df_drifted)
    assert result["threshold_exceeded"]
    assert result["status"] == "drifting"
    assert result["drift_score"] > 2.0
