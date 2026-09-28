"""Shared, cached access to the PMSM replay dataset.

Loaded once per process (it is ~1.3M rows) and annotated with the same proxy
fault labels the model was trained against, so simulated predictions can be
compared with ground truth.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from voltguard.features.engine import FEATURE_COLUMNS, generate_fault_codes

DEFAULT_DATA_PATH = "data/comprehensive_fault_training_data.csv"


def data_path() -> Path:
    return Path(os.getenv("VOLTGUARD_REPLAY_DATA", DEFAULT_DATA_PATH))


@lru_cache(maxsize=1)
def load_replay_data() -> pd.DataFrame | None:
    """Return the replay dataset with a ``true_fault`` column, or None if absent."""
    path = data_path()
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["true_fault"] = generate_fault_codes(df)["fault_code"].astype(int)
    return df


@lru_cache(maxsize=1)
def stress_start_indices() -> np.ndarray:
    """Row indices just before high-load regions, so replays show transitions."""
    df = load_replay_data()
    if df is None:
        return np.array([], dtype=int)
    stressed = (df["i_q"].abs() > df["i_q"].quantile(0.85)) | (
        df["coolant"] > df["coolant"].quantile(0.85)
    )
    return np.maximum(0, np.flatnonzero(stressed.to_numpy()) - 50)


@lru_cache(maxsize=1)
def feature_ranges() -> dict[str, dict[str, float]] | None:
    """2nd/50th/98th percentiles of raw model inputs, for scenario sliders."""
    df = load_replay_data()
    if df is None:
        return None
    q = df[FEATURE_COLUMNS].quantile([0.02, 0.5, 0.98])
    return {
        col: {
            "min": float(q.at[0.02, col]),
            "median": float(q.at[0.5, col]),
            "max": float(q.at[0.98, col]),
        }
        for col in FEATURE_COLUMNS
    }


@lru_cache(maxsize=1)
def profile_summary() -> list[dict] | None:
    """Per-drive row count and proxy-fault rate."""
    df = load_replay_data()
    if df is None:
        return None
    g = df.groupby("profile_id")["true_fault"]
    out = pd.DataFrame({"rows": g.size(), "fault_rate": g.apply(lambda s: float((s != 0).mean()))})
    return [
        {"profile_id": int(pid), "rows": int(r.rows), "fault_rate": round(r.fault_rate, 4)}
        for pid, r in out.iterrows()
    ]
