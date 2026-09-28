"""Data drift detection logic.

Compares incoming live batch distributions against a baseline captured at training time.
Includes compatibility repair for legacy baseline artifacts.
"""

from __future__ import annotations

import logging

import pandas as pd

from voltguard.diagnostics.registry import ModelRegistry
from voltguard.features.engine import ALL_FEATURES, engineer_features
from voltguard.simulator.dataset import data_path as replay_data_path

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 2.0


class DriftDetector:
    """Detects feature drift using baseline z-score distances."""

    def __init__(self, version: str = "latest"):
        self.registry = ModelRegistry()
        _, _, _, baseline = self.registry.load(version)
        self.version = version
        self.baseline = baseline or {}
        self._ensure_compatible_baseline()

        if not self.baseline:
            logger.warning(
                "No baseline found for '%s'. Drift scoring will return no_baseline.",
                version,
            )

    def _ensure_compatible_baseline(self) -> None:
        """Repair legacy baselines that don't match ALL_FEATURES."""
        if not self.baseline:
            self.baseline = self._build_baseline_from_training_data()
            return

        missing = [col for col in ALL_FEATURES if col not in self.baseline]
        if not missing:
            return

        rebuilt = self._build_baseline_from_training_data()
        if not rebuilt:
            logger.warning(
                "Baseline missing features %s and auto-rebuild failed.",
                missing,
            )
            return

        for col in missing:
            self.baseline[col] = rebuilt[col]

        logger.warning(
            "Baseline for '%s' was legacy/incompatible. Filled missing features: %s",
            self.version,
            missing,
        )

    def _build_baseline_from_training_data(self) -> dict:
        """Build baseline directly from local training CSV if available."""
        data_path = replay_data_path()
        if not data_path.exists():
            return {}

        df = pd.read_csv(data_path)
        df = engineer_features(df)

        if not set(ALL_FEATURES).issubset(df.columns):
            return {}

        baseline = {}
        for col in ALL_FEATURES:
            mean_val = float(df[col].mean())
            std_val = float(df[col].std())
            baseline[col] = {"mean": mean_val, "std": std_val}
        return baseline

    def calculate_drift_score(self, df_incoming: pd.DataFrame) -> dict:
        """Compute average z-score drift against baseline."""
        if not self.baseline:
            return {"drift_score": 0.0, "status": "no_baseline", "features": {}}

        df = engineer_features(df_incoming.copy())

        available = [col for col in ALL_FEATURES if col in df.columns and col in self.baseline]
        if not available:
            return {
                "drift_score": 0.0,
                "threshold_exceeded": False,
                "status": "no_compatible_features",
                "features": {},
            }

        feature_scores: dict[str, float] = {}
        total_drift = 0.0

        for col in available:
            base_mean = float(self.baseline[col].get("mean", 0.0))
            base_std = float(self.baseline[col].get("std", 0.0))
            if base_std == 0.0:
                base_std = 1e-6

            new_mean = float(df[col].mean())
            z_score = abs((new_mean - base_mean) / base_std)
            feature_scores[col] = z_score
            total_drift += z_score

        avg_score = float(total_drift / len(available))

        return {
            "drift_score": avg_score,
            "threshold_exceeded": avg_score > DRIFT_THRESHOLD,
            "status": "drifting" if avg_score > DRIFT_THRESHOLD else "nominal",
            "features": feature_scores,
            "features_used": len(available),
        }
