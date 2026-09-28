"""Predictive forecasting engine.

Predicts whether a fault is likely to occur in the next N ticks by
extrapolating recent numeric trends.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from voltguard.diagnostics.predictor import FaultPredictor
from voltguard.features.engine import ALL_FEATURES, FAULT_LABELS, engineer_features

logger = logging.getLogger(__name__)


class FaultForecaster:
    """Forecasts impending faults before they fully manifest."""

    def __init__(self, predictor_version: str = "latest", forecast_horizon_ticks: int = 5):
        self.predictor = FaultPredictor(version=predictor_version)
        self.forecast_horizon = forecast_horizon_ticks
        logger.info(
            "Initialized FaultForecaster with horizon=%d ticks using predictor version='%s'",
            self.forecast_horizon,
            predictor_version,
        )

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        """Convert a value to float, returning None when invalid."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def forecast(self, vehicle_id: str, current_input: dict[str, float]) -> dict[str, Any]:
        """Extrapolate sensor states and predict future fault probability."""
        cleaned_input = {
            k: v
            for k, v in current_input.items()
            if k != "vehicle_id" and self._to_float_or_none(v) is not None
        }

        current_prediction = self.predictor.predict(vehicle_id, cleaned_input)

        history_df = self.predictor.history.get(vehicle_id)
        if history_df is None or len(history_df) < 3:
            return {
                "forecasted_fault": 0,
                "confidence": 0.0,
                "horizon_ticks": self.forecast_horizon,
                "warning": "Insufficient history for forecasting.",
            }

        recent = history_df.tail(3)
        numeric_keys = [k for k in cleaned_input if k in recent.columns]

        if not numeric_keys:
            return {
                "forecasted_fault": current_prediction["predicted_fault"],
                "fault_label": FAULT_LABELS.get(current_prediction["predicted_fault"], "Unknown"),
                "confidence": float(current_prediction["confidence"]),
                "horizon_ticks": self.forecast_horizon,
                "warning": "No numeric keys available for extrapolation.",
                "current_prediction": current_prediction["predicted_fault"],
            }

        rates = (recent.iloc[-1][numeric_keys] - recent.iloc[0][numeric_keys]) / 2.0

        projected_input: dict[str, float] = {}
        for key in numeric_keys:
            current_val = self._to_float_or_none(cleaned_input.get(key))
            if current_val is None:
                continue
            delta = float(rates.get(key, 0.0)) * self.forecast_horizon
            projected_input[key] = current_val + delta

        # Keep non-numeric or absent values from current tick as-is where possible.
        for key, value in cleaned_input.items():
            if key not in projected_input:
                converted = self._to_float_or_none(value)
                if converted is not None:
                    projected_input[key] = converted

        future_df = pd.concat([history_df, pd.DataFrame([projected_input])], ignore_index=True)
        future_df["vehicle_id"] = vehicle_id

        enriched_future = engineer_features(future_df)
        latest_future_features = enriched_future.iloc[[-1]]

        feature_values = latest_future_features[ALL_FEATURES].fillna(0)
        scaled_future = self.predictor.scaler.transform(feature_values)

        pred = int(self.predictor.model.predict(scaled_future)[0])
        proba = self.predictor.model.predict_proba(scaled_future)[0]
        conf = float(max(proba))

        extrapolated_values = {
            "stator_winding_projected": float(projected_input.get("stator_winding", 0.0)),
            "coolant_projected": float(projected_input.get("coolant", 0.0)),
            "motor_speed_projected": float(projected_input.get("motor_speed", 0.0)),
            "battery_temp_projected": float(projected_input.get("battery_temp", 0.0)),
        }

        return {
            "forecasted_fault": pred,
            "fault_label": FAULT_LABELS.get(pred, "Unknown"),
            "confidence": conf,
            "horizon_ticks": self.forecast_horizon,
            "projected_state": extrapolated_values,
            "current_prediction": current_prediction["predicted_fault"],
        }
