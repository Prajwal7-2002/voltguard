from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.shared import apply_theme, class_name, render_page_header
from voltguard.diagnostics.predictor import FaultPredictor
from voltguard.simulator.vehicle import VehicleSimulator

st.set_page_config(page_title="Fleet Overview", page_icon="VG", layout="wide")
apply_theme()
render_page_header(
    "Fleet Overview",
    "Multi-vehicle operational view with risk concentration and unstable unit ranking.",
)

FLEET_SIZE = 8

@st.cache_resource(show_spinner=False)
def init_fleet() -> tuple[list[VehicleSimulator], FaultPredictor]:
    vehicles = [VehicleSimulator(f"v-{i+1:03d}") for i in range(FLEET_SIZE)]
    predictor = FaultPredictor("latest")
    return vehicles, predictor

try:
    vehicles, predictor = init_fleet()
except Exception as exc:
    st.error(f"Unable to initialize fleet stack: {exc}")
    st.stop()

if "fleet_tick" not in st.session_state:
    st.session_state.fleet_tick = 0
if "fleet_rows" not in st.session_state:
    st.session_state.fleet_rows = []

c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
rounds = c1.slider("Rounds to run", 1, 30, 5, 1)
run_round = c2.button("Run fleet rounds", use_container_width=True)
if c3.button("Reset fleet session", use_container_width=True):
    st.session_state.fleet_tick = 0
    st.session_state.fleet_rows = []
    st.rerun()

if run_round:
    progress = st.progress(0, text="Processing fleet rounds...")
    for r in range(rounds):
        st.session_state.fleet_tick += 1
        current_tick = st.session_state.fleet_tick
        for vehicle in vehicles:
            reading = vehicle.tick()
            vehicle_id = reading.get("vehicle_id", vehicle.vehicle_id)
            model_input = {k: v for k, v in reading.items() if k != "vehicle_id"}
            pred = predictor.predict(vehicle_id, model_input)

            fault_code = int(pred.get("predicted_fault", 0))
            confidence = float(pred.get("confidence", 0.0))
            nominal_prob = float(pred.get("probabilities", {}).get(0, 0.0))
            risk = (1.0 - nominal_prob) * 100.0
            health_score = max(1.0, 100.0 - risk)

            st.session_state.fleet_rows.append(
                {
                    "tick": current_tick,
                    "vehicle_id": vehicle_id,
                    "fault_code": fault_code,
                    "fault_label": class_name(fault_code),
                    "confidence": confidence,
                    "risk": risk,
                    "health_score": health_score,
                    "coolant": float(model_input.get("coolant", 0.0)),
                    "battery_temp": float(model_input.get("battery_temp", 0.0)),
                    "motor_speed": float(model_input.get("motor_speed", 0.0)),
                    "soc": float(model_input.get("soc", 0.0)),
                }
            )
        progress.progress((r + 1) / rounds, text=f"Round {r + 1}/{rounds}")
    progress.empty()

fleet_df = pd.DataFrame(st.session_state.fleet_rows)
if fleet_df.empty:
    st.info("Run fleet rounds to populate this page.")
    st.stop()

latest_tick = int(fleet_df["tick"].max())
latest = fleet_df[fleet_df["tick"] == latest_tick].copy()

healthy = int((latest["fault_code"] == 0).sum())
at_risk = int((latest["fault_code"] != 0).sum())
mean_risk = float(latest["risk"].mean())
mean_health = float(latest["health_score"].mean())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Current Tick", str(latest_tick))
k2.metric("Fleet Size", str(FLEET_SIZE))
k3.metric("Healthy Units", str(healthy))
k4.metric("At-Risk Units", str(at_risk))
k5.metric("Avg Risk", f"{mean_risk:.1f}%")

left, right = st.columns([1.35, 1.0])
with left:
    st.markdown("### Latest Fleet Snapshot")
    view_cols = [
        "vehicle_id",
        "fault_label",
        "confidence",
        "risk",
        "health_score",
        "coolant",
        "battery_temp",
        "motor_speed",
        "soc",
    ]
    st.dataframe(
        latest[view_cols].sort_values("risk", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.markdown("### Current Fault Distribution")
    mix = latest["fault_label"].value_counts().rename_axis("fault").reset_index(name="count")
    fig_mix = px.pie(mix, names="fault", values="count", hole=0.45)
    fig_mix.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_mix, use_container_width=True)

st.markdown("### Fleet Risk Heatmap (Recent 30 Ticks)")
recent = fleet_df[fleet_df["tick"] >= max(1, latest_tick - 29)].copy()
pivot = recent.pivot_table(index="vehicle_id", columns="tick", values="risk", aggfunc="mean")
if not pivot.empty:
    heatmap_df = pivot.reset_index().melt(id_vars="vehicle_id", var_name="tick", value_name="risk")
    fig_heat = px.density_heatmap(
        heatmap_df,
        x="tick",
        y="vehicle_id",
        z="risk",
        color_continuous_scale="YlOrRd",
        template="plotly_white",
    )
    fig_heat.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("### Most Unstable Vehicles")
vehicle_rank = (
    recent.groupby("vehicle_id")
    .agg(avg_risk=("risk", "mean"), fault_events=("fault_code", lambda s: int((s != 0).sum())))
    .reset_index()
    .sort_values(["avg_risk", "fault_events"], ascending=False)
)
st.dataframe(vehicle_rank, use_container_width=True, hide_index=True)

st.markdown("### Risk Trend")
trend = recent.groupby("tick").agg(avg_risk=("risk", "mean")).reset_index()
fig_trend = px.line(trend, x="tick", y="avg_risk", template="plotly_white")
fig_trend.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_trend, use_container_width=True)
