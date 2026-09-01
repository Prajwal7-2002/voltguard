from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.shared import apply_theme, class_name, render_page_header
from voltguard.features.engine import FEATURE_COLUMNS, engineer_features, generate_fault_codes

st.set_page_config(page_title="Historical", page_icon="VG", layout="wide")
apply_theme()
render_page_header(
    "Historical Analytics",
    "Distribution and pattern analysis over sampled training telemetry with engineered fault labels.",
)

DATA_PATH = Path("data/comprehensive_fault_training_data.csv")

@st.cache_data(show_spinner=False)
def load_sample(sample_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    sample = engineer_features(sample)
    sample = generate_fault_codes(sample)
    return sample

if not DATA_PATH.exists():
    st.error("Dataset not found: data/comprehensive_fault_training_data.csv")
    st.stop()

side1, side2 = st.columns([1.0, 1.0])
sample_size = side1.slider("Sample size", min_value=2000, max_value=50000, value=8000, step=2000)
seed = side2.number_input("Sampling seed", min_value=1, max_value=9999, value=42)

df = load_sample(sample_size, int(seed))

fault_rate = float((df["fault_code"] != 0).mean() * 100.0)
nominal = int((df["fault_code"] == 0).sum())
non_nominal = int((df["fault_code"] != 0).sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Samples", f"{len(df):,}")
k2.metric("Fault Rate", f"{fault_rate:.2f}%")
k3.metric("Nominal Rows", f"{nominal:,}")
k4.metric("Fault Rows", f"{non_nominal:,}")

left, right = st.columns([1.1, 1.0])

with left:
    st.markdown("### Fault Class Distribution")
    counts = df["fault_code"].value_counts().sort_index()
    dist_df = pd.DataFrame(
        {
            "fault": [class_name(int(c)) for c in counts.index],
            "count": counts.values,
        }
    )
    fig_dist = px.bar(dist_df, x="fault", y="count", template="plotly_white")
    fig_dist.update_layout(height=290, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_dist, use_container_width=True)

with right:
    st.markdown("### Fault Burden by Driving Profile")
    if "profile_id" in df.columns:
        profile_df = (
            df.groupby("profile_id")
            .agg(fault_rate=("fault_code", lambda s: float((s != 0).mean() * 100.0)), rows=("fault_code", "count"))
            .reset_index()
            .sort_values("fault_rate", ascending=False)
        )
        fig_profile = px.bar(profile_df, x="profile_id", y="fault_rate", template="plotly_white")
        fig_profile.update_layout(height=290, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_profile, use_container_width=True)
    else:
        st.info("profile_id column is not available in sampled data.")

st.markdown("### Sensor Behavior by Fault Class")
feature_candidates = [c for c in FEATURE_COLUMNS + ["battery_temp", "power_draw", "thermal_delta"] if c in df.columns]
feature_name = st.selectbox("Feature", feature_candidates, index=min(1, len(feature_candidates) - 1))

plot_df = df[["fault_code", feature_name]].dropna().copy()
plot_df["fault"] = plot_df["fault_code"].map(lambda c: class_name(int(c)))
fig_box = px.box(plot_df, x="fault", y=feature_name, points=False, template="plotly_white")
fig_box.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("### Correlation Lens (Operational Features)")
focus = [c for c in FEATURE_COLUMNS + ["battery_temp", "power_draw", "thermal_delta", "dt_dt"] if c in df.columns]
if len(focus) >= 2:
    corr = df[focus].corr(numeric_only=True).reset_index().melt(id_vars="index", var_name="feature", value_name="corr")
    fig_corr = px.density_heatmap(
        corr,
        x="index",
        y="feature",
        z="corr",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        template="plotly_white",
    )
    fig_corr.update_layout(height=420, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("### Statistical Summary")
summary = (
    df.groupby("fault_code")[feature_candidates]
    .mean(numeric_only=True)
    .rename(index=lambda c: class_name(int(c)))
    .round(3)
)
st.dataframe(summary, use_container_width=True)
