from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.shared import apply_theme, class_name, image_if_exists, render_page_header, top_n_explanations
from voltguard.diagnostics.predictor import FaultPredictor
from voltguard.features.engine import FEATURE_COLUMNS

st.set_page_config(page_title="Explainability", page_icon="VG", layout="wide")
apply_theme()
render_page_header(
    "Explainability Workbench",
    "Inspect model rationale using class probabilities and local SHAP evidence on custom sensor scenarios.",
)

@st.cache_resource(show_spinner=False)
def get_predictor() -> FaultPredictor:
    return FaultPredictor("latest")

@st.cache_data(show_spinner=False)
def load_bounds() -> dict[str, tuple[float, float, float]]:
    path = Path("data/comprehensive_fault_training_data.csv")
    if not path.exists():
        return {
            "ambient": (0.0, 50.0, 30.0),
            "coolant": (0.0, 100.0, 45.0),
            "u_d": (-4.0, 4.0, 0.0),
            "u_q": (-4.0, 4.0, 0.0),
            "motor_speed": (0.0, 12000.0, 3000.0),
            "torque": (-300.0, 300.0, 0.0),
            "i_d": (-300.0, 300.0, 0.0),
            "i_q": (-300.0, 300.0, 0.0),
        }

    df = pd.read_csv(path, usecols=[c for c in FEATURE_COLUMNS if c in pd.read_csv(path, nrows=0).columns])
    bounds: dict[str, tuple[float, float, float]] = {}
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            bounds[col] = (0.0, 1.0, 0.0)
            continue
        lo = float(s.quantile(0.02))
        hi = float(s.quantile(0.98))
        mid = float(s.median())
        bounds[col] = (lo, hi, mid)
    return bounds

try:
    predictor = get_predictor()
except Exception as exc:
    st.error(f"Model unavailable: {exc}")
    st.stop()

bounds = load_bounds()

st.markdown("### Scenario Builder")
controls = st.columns(4)
input_data: dict[str, float] = {}
for idx, col in enumerate(FEATURE_COLUMNS):
    lo, hi, mid = bounds.get(col, (0.0, 1.0, 0.0))
    if lo == hi:
        hi = lo + 1.0
    with controls[idx % 4]:
        input_data[col] = st.slider(col, min_value=float(lo), max_value=float(hi), value=float(mid), step=(hi - lo) / 200.0)

vehicle_token = st.text_input("Scenario ID", value="xai-sandbox")
run = st.button("Run explainability", use_container_width=True)

if run:
    result = predictor.predict(vehicle_token, input_data)

    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted Fault", class_name(int(result["predicted_fault"])))
    k2.metric("Confidence", f"{float(result['confidence']):.4f}")
    risk = (1.0 - float(result.get("probabilities", {}).get(0, 0.0))) * 100.0
    k3.metric("Risk vs Nominal", f"{risk:.1f}%")

    left, right = st.columns([1.0, 1.0])

    with left:
        st.markdown("### Class Probability Profile")
        prob_df = pd.DataFrame(
            {
                "fault": [class_name(int(k)) for k in result["probabilities"].keys()],
                "probability": [float(v) for v in result["probabilities"].values()],
            }
        ).sort_values("probability", ascending=False)
        fig_prob = px.bar(prob_df, x="fault", y="probability", template="plotly_white")
        fig_prob.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_prob, use_container_width=True)

    with right:
        st.markdown("### Local SHAP Contributions")
        shap_df = top_n_explanations(result.get("explanations", {}), n=12)
        if not shap_df.empty:
            fig_shap = px.bar(
                shap_df.sort_values("contribution", ascending=True),
                x="contribution",
                y="feature",
                orientation="h",
                template="plotly_white",
                color="contribution",
                color_continuous_scale="RdBu",
            )
            fig_shap.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("### Interpretation Guide")
    st.write("- Positive SHAP value means that feature pushes toward the predicted class.")
    st.write("- Negative SHAP value means that feature pushes away from the predicted class.")
    st.write("- Confidence alone is not enough. Use dominant features plus probability spread.")

st.markdown("---")
st.markdown("### Global Explainability Artifacts")
image_if_exists("results/shap_summary.png", "Global SHAP summary from evaluation pipeline")
