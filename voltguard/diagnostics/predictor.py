"""Real-time fault prediction with SHAP explainability.

Loads a trained model and scaler, produces predictions with per-feature
SHAP explanations. Maintains a rolling history buffer to compute
time-series (lag) features on the fly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from voltguard.core.logging import get_logger
from voltguard.features.engine import ALL_FEATURES, engineer_features, FAULT_LABELS
from voltguard.diagnostics.registry import ModelRegistry

logger = get_logger(__name__)


class FaultPredictor:
    """Encapsulates model loading, feature computation, and SHAP.

    Usage:
        predictor = FaultPredictor("latest")
        result = predictor.predict("v1", {"ambient": 30, "coolant": 45, ...})
    """

    def __init__(self, version: str = "latest") -> None:
        self.registry = ModelRegistry()
        self.model, self.scaler, self.metrics, self.baseline = self.registry.load(version)
        import shap  # heavy import (numba); defer until a model is actually loaded

        self.explainer = shap.TreeExplainer(self.model)

        # Dictionary mapping vehicle_id -> rolling DataFrame history
        self.history: dict[str, pd.DataFrame] = {}
        self.max_history = 10  # Enough to compute rolling features

        logger.info("Loaded 5-class powertrain model (version: %s).", version)

    def predict(self, vehicle_id: str, input_data: dict[str, float]) -> dict:
        """Run fault prediction on a single sensor reading.

        Args:
            vehicle_id: Identifier for the vehicle (needed for history buffer).
            input_data: Dictionary of raw sensor values.

        Returns:
            Dictionary with predicted_fault, confidence, explanations, and input.
        """
        # Update history buffer
        df_new = pd.DataFrame([input_data])
        df_new["vehicle_id"] = vehicle_id

        if vehicle_id not in self.history:
            self.history[vehicle_id] = df_new
        else:
            self.history[vehicle_id] = pd.concat(
                [self.history[vehicle_id], df_new], ignore_index=True
            ).tail(self.max_history)

        # Compute features using the full history buffer
        enriched_history = engineer_features(self.history[vehicle_id])

        # The latest row is the current state with fully computed features
        latest_features = enriched_history.iloc[[-1]]

        # Use ALL_FEATURES (leak-proof base + engineered columns)
        feature_values = latest_features[ALL_FEATURES].fillna(0)
        scaled = self.scaler.transform(feature_values)

        # Prediction
        pred = int(self.model.predict(scaled)[0])
        proba = self.model.predict_proba(scaled)[0]
        conf = float(max(proba))

        # SHAP explanations
        shap_values = self.explainer.shap_values(scaled)

        if isinstance(shap_values, list):
            class_shap = np.asarray(shap_values[pred]).flatten()
        else:
            class_shap = np.asarray(shap_values).flatten()

        # Map SHAP values to feature names
        explanations = {}
        for i, feat in enumerate(ALL_FEATURES):
            if i < len(class_shap):
                explanations[feat] = float(class_shap[i])

        return {
            "predicted_fault": pred,
            "fault_label": FAULT_LABELS.get(pred, "Unknown"),
            "confidence": conf,
            "probabilities": {i: float(p) for i, p in enumerate(proba)},
            "explanations": explanations,
            "input": input_data,
        }


# ---------------------------------------------------------------------------
# Convenience function (backward-compatible)
# ---------------------------------------------------------------------------

def load_model_and_scaler(version: str = "latest"):
    """Legacy loader — prefer FaultPredictor class instead."""
    registry = ModelRegistry()
    model, scaler, _, _ = registry.load(version)
    return model, scaler
