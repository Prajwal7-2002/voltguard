"""Shared dashboard helpers for styling, model context, and display utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from voltguard.diagnostics.registry import ModelRegistry
from voltguard.features.engine import FAULT_LABELS


def apply_theme() -> None:
    """Apply a consistent visual system across pages."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: "Space Grotesk", sans-serif;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.0rem;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}

.vg-card {
    border: 1px solid #d4dde9;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    background: linear-gradient(180deg, #fbfdff 0%, #f1f6fb 100%);
}

.vg-card h4 {
    margin: 0;
    font-size: 0.9rem;
    color: #3d4d60;
}

.vg-card p {
    margin: 0.25rem 0 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: #0d253d;
}

.vg-note {
    border-left: 4px solid #0a6fb4;
    background: #eef7ff;
    padding: 0.65rem 0.8rem;
    border-radius: 8px;
    color: #16344e;
    font-size: 0.92rem;
}

.stMetric {
    border: 1px solid #d4dde9;
    border-radius: 12px;
    padding: 0.5rem 0.5rem 0.5rem 0.5rem;
    background: #ffffff;
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_model_snapshot() -> dict[str, Any]:
    """Load latest registry metadata and metrics."""
    registry = ModelRegistry()
    latest = registry.get_latest_version()
    _, _, metrics, baseline = registry.load(latest)
    return {"version": latest, "metrics": metrics, "baseline": baseline}


def metric_value(metrics: dict[str, Any], key: str, fallback_keys: list[str] | None = None) -> float:
    """Return a numeric metric value using fallback keys if needed."""
    if key in metrics:
        return float(metrics[key])
    for alt in fallback_keys or []:
        if alt in metrics:
            return float(metrics[alt])
    return 0.0


def class_name(code: int) -> str:
    """Map fault code to display label."""
    return FAULT_LABELS.get(code, f"Code {code}")


def find_metrics_per_class(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Normalize per-class metrics regardless of key naming."""
    per_class = metrics.get("per_class_report")
    if isinstance(per_class, dict) and per_class:
        return per_class

    alt = metrics.get("per_class")
    if isinstance(alt, dict) and alt:
        return alt
    return {}


def image_if_exists(path: str | Path, caption: str) -> None:
    """Render an image only if artifact exists."""
    p = Path(path)
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing artifact: {p}. Retrain to regenerate.")


def top_n_explanations(explanations: dict[str, float], n: int = 8) -> pd.DataFrame:
    """Return top-N absolute SHAP contributions as a dataframe."""
    if not explanations:
        return pd.DataFrame(columns=["feature", "contribution"])
    df = pd.DataFrame(
        [{"feature": k, "contribution": float(v), "abs_value": abs(float(v))} for k, v in explanations.items()]
    )
    return df.sort_values("abs_value", ascending=False).head(n).drop(columns=["abs_value"])


def render_page_header(title: str, subtitle: str) -> None:
    """Consistent page heading."""
    st.title(title)
    st.caption(subtitle)
