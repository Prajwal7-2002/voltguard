"""Model registry for ML artifact versioning and retrieval."""

import json
import re
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Tuple, Optional

import joblib

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Handles versioning, saving, and loading of ML models."""

    def __init__(self, base_dir: str | Path = "models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.latest_pointer = self.base_dir / "latest.txt"

    def _get_next_version(self) -> str:
        """Determines the next version string (e.g., 'v1', 'v2')."""
        versions = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    num = int(d.name[1:])
                    versions.append(num)
                except ValueError:
                    pass
        next_num = max(versions) + 1 if versions else 1
        return f"v{next_num}"

    def save(
        self,
        model: Any,
        scaler: Any,
        metrics: dict,
        version: Optional[str] = None,
        baseline: Optional[dict] = None
    ) -> str:
        """Saves model, scaler, metrics, and data baselines under a specific version."""
        if version is None:
            version = self._get_next_version()

        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, version_dir / "model.pkl")
        joblib.dump(scaler, version_dir / "scaler.pkl")
        
        with open(version_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        if baseline is not None:
            with open(version_dir / "baseline.json", "w") as f:
                json.dump(baseline, f, indent=2)

        # Update the latest pointer instead of relying on symlinks (for Windows compatibility)
        with open(self.latest_pointer, "w") as f:
            f.write(version)

        logger.info(f"Saved model artifacts to {version_dir}")
        return version

    def get_latest_version(self) -> str:
        """Retrieves the string of the latest model version."""
        if not self.latest_pointer.exists():
            raise FileNotFoundError("No models found in the registry.")
        with open(self.latest_pointer, "r") as f:
            return f.read().strip()

    def load(self, version: str = "latest") -> Tuple[Any, Any, dict, Optional[dict]]:
        """Loads model, scaler, and metrics given a version string (or 'latest')."""
        if version == "latest":
            version = self.get_latest_version()
        if not re.fullmatch(r"v\d+", version):
            raise FileNotFoundError(f"Invalid model version {version!r}")

        version_dir = self.base_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found at {version_dir}")

        model = joblib.load(version_dir / "model.pkl")
        scaler = joblib.load(version_dir / "scaler.pkl")
        
        with open(version_dir / "metrics.json", "r") as f:
            metrics = json.load(f)
            
        baseline = None
        if (version_dir / "baseline.json").exists():
            with open(version_dir / "baseline.json", "r") as f:
                baseline = json.load(f)

        logger.info(f"Loaded model artifacts from {version_dir}")
        return model, scaler, metrics, baseline
