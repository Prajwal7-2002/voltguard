from __future__ import annotations

import os
import sys
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.shared import apply_theme, class_name, render_page_header, top_n_explanations
from voltguard.diagnostics.predictor import FaultPredictor
from voltguard.root_cause.analyzer import resolve_fault
from voltguard.simulator.vehicle import VehicleSimulator


def risk_band(risk_value: float) -> str:
    """Map risk score to operator-friendly severity band."""
    if risk_value >= 80:
        return "Critical"
    if risk_value >= 55:
        return "High"
    if risk_value >= 30:
        return "Moderate"
    return "Low"


st.set_page_config(page_title="Live Monitor", page_icon="VG", layout="wide")
apply_theme()
render_page_header(
    "Live Monitor",
    "Tick-level inference view with risk score, class probabilities, SHAP evidence, and actions.",
)

with st.expander("How to read this page", expanded=False):
    st.write("1. Run ticks to generate live telemetry and model predictions.")
    st.write("2. Start with Current Diagnosis to see fault, risk, and recommended action.")
    st.write("3. Use Class Probabilities to check uncertainty, then SHAP Drivers for why.")
    st.write("4. Use timeline and fault mix to spot persistent or repeating failure modes.")

if "live_vehicle" not in st.session_state:
    st.session_state.live_vehicle = VehicleSimulator("v-001")
if "live_predictor" not in st.session_state:
    try:
        st.session_state.live_predictor = FaultPredictor("latest")
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        st.stop()
if "live_tick" not in st.session_state:
    st.session_state.live_tick = 0
if "live_rows" not in st.session_state:
    st.session_state.live_rows = []
if "live_events" not in st.session_state:
    st.session_state.live_events = []
if "live_latest" not in st.session_state:
    st.session_state.live_latest = {}
if "live_last_pred" not in st.session_state:
    st.session_state.live_last_pred = {}

left, mid, right = st.columns([1.1, 1.1, 1.8])
step_count = left.slider("Ticks to run", 1, 200, 25, 1)
run_now = mid.button("Run ticks", use_container_width=True)
if right.button("Reset session", use_container_width=True):
    st.session_state.live_vehicle = VehicleSimulator("v-001")
    st.session_state.live_tick = 0
    st.session_state.live_rows = []
    st.session_state.live_events = []
    st.session_state.live_latest = {}
    st.session_state.live_last_pred = {}
    st.rerun()

if run_now:
    progress = st.progress(0, text="Running live ticks...")
    for i in range(step_count):
        st.session_state.live_tick += 1
        tick_id = st.session_state.live_tick

        reading = st.session_state.live_vehicle.tick()
        vehicle_id = reading.get("vehicle_id", "v-001")
        model_input = {k: v for k, v in reading.items() if k != "vehicle_id"}

        prediction = st.session_state.live_predictor.predict(vehicle_id, model_input)
        st.session_state.live_latest = model_input
        st.session_state.live_last_pred = prediction

        probs = prediction.get("probabilities", {})
        nominal_prob = float(probs.get(0, 0.0))
        risk_score = (1.0 - nominal_prob) * 100.0
        fault_code = int(prediction.get("predicted_fault", 0))
        confidence = float(prediction.get("confidence", 0.0))

        explanations = prediction.get("explanations", {})
        dominant_feature = "n/a"
        if explanations:
            dominant_feature = max(explanations, key=lambda x: abs(float(explanations[x])))

        root_cause = "No fault"
        action = "No action"
        if fault_code != 0:
            root_cause, action = resolve_fault(fault_code, explanations, model_input)
            st.session_state.live_vehicle.apply_effect(action)

        st.session_state.live_rows.append(
            {
                "tick": tick_id,
                "fault_code": fault_code,
                "fault_label": class_name(fault_code),
                "confidence": confidence,
                "risk_score": risk_score,
                "coolant": float(model_input.get("coolant", 0.0)),
                "battery_temp": float(model_input.get("battery_temp", 0.0)),
                "motor_speed": float(model_input.get("motor_speed", 0.0)),
                "soc": float(model_input.get("soc", 0.0)),
            }
        )

        if fault_code != 0:
            st.session_state.live_events.insert(
                0,
                {
                    "tick": tick_id,
                    "fault": class_name(fault_code),
                    "confidence": round(confidence, 4),
                    "root_cause": root_cause,
                    "dominant_feature": dominant_feature,
                    "action": action,
                },
            )

        progress.progress((i + 1) / step_count, text=f"Tick {i + 1}/{step_count}")

    st.session_state.live_rows = st.session_state.live_rows[-1500:]
    st.session_state.live_events = st.session_state.live_events[:300]
    progress.empty()

rows_df = pd.DataFrame(st.session_state.live_rows)
last_pred = st.session_state.live_last_pred

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Ticks", str(st.session_state.live_tick))
c2.metric("Current Fault", class_name(int(last_pred.get("predicted_fault", 0))) if last_pred else "n/a")
c3.metric("Current Confidence", f"{float(last_pred.get('confidence', 0.0)):.3f}" if last_pred else "n/a")
if last_pred:
    risk = (1.0 - float(last_pred.get("probabilities", {}).get(0, 0.0))) * 100.0
else:
    risk = 0.0
c4.metric("Current Risk", f"{risk:.1f}%")
c5.metric("Fault Events", str(len(st.session_state.live_events)))

if rows_df.empty:
    st.info("Run ticks to populate live diagnostics.")
    st.stop()

latest_input = st.session_state.live_latest
last_fault_code = int(last_pred.get("predicted_fault", 0)) if last_pred else 0
last_confidence = float(last_pred.get("confidence", 0.0)) if last_pred else 0.0
latest_explanations = last_pred.get("explanations", {}) if last_pred else {}
dominant_feature = "n/a"
if latest_explanations:
    dominant_feature = max(latest_explanations, key=lambda x: abs(float(latest_explanations[x])))

if last_fault_code != 0 and latest_input:
    summary_root, summary_action = resolve_fault(last_fault_code, latest_explanations, latest_input)
else:
    summary_root, summary_action = ("No active fault", "No action required")

st.markdown("### Current Diagnosis")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Risk Band", risk_band(risk))
s2.metric("Predicted Fault", class_name(last_fault_code))
s3.metric("Confidence", f"{last_confidence:.3f}")
s4.metric("Dominant Driver", dominant_feature)
st.caption(f"Root cause: {summary_root}")
st.caption(f"Recommended action: {summary_action}")

left_panel, right_panel = st.columns([1.55, 1.0])

with left_panel:
    st.markdown("### Trend Timeline")
    timeline = rows_df[["tick", "risk_score", "coolant", "battery_temp", "motor_speed", "soc"]].copy()
    metric_options = {
        "Risk Score (%)": "risk_score",
        "Coolant Temp (C)": "coolant",
        "Battery Temp (C)": "battery_temp",
        "Motor Speed (RPM)": "motor_speed",
        "State of Charge (%)": "soc",
    }
    metric_label = st.selectbox(
        "Select timeline signal",
        list(metric_options.keys()),
        index=0,
    )
    metric_name = metric_options[metric_label]
    fig_line = px.line(timeline.tail(300), x="tick", y=metric_name, template="plotly_white")
    fig_line.update_yaxes(title=metric_label)
    fig_line.update_layout(height=310, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("### Fault Mix (Last 300 Ticks)")
    recent_labels = rows_df.tail(300)["fault_label"].tolist()
    class_counter = Counter(recent_labels)
    mix_df = pd.DataFrame({"fault": list(class_counter.keys()), "count": list(class_counter.values())})
    fig_mix = px.bar(mix_df.sort_values("count", ascending=False), x="fault", y="count", template="plotly_white")
    fig_mix.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_mix, use_container_width=True)

with right_panel:
    st.markdown("### Current Class Probabilities")
    prob_map = last_pred.get("probabilities", {}) if last_pred else {}
    if prob_map:
        prob_df = pd.DataFrame(
            {
                "fault": [class_name(int(code)) for code in prob_map.keys()],
                "probability": [float(val) for val in prob_map.values()],
            }
        ).sort_values("probability", ascending=False)
        fig_prob = px.bar(prob_df, x="probability", y="fault", orientation="h", template="plotly_white")
        fig_prob.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.info("No probability output yet.")

    st.markdown("### Top SHAP Drivers")
    shap_df = top_n_explanations(last_pred.get("explanations", {}) if last_pred else {}, n=10)
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
        fig_shap.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("No SHAP explanation available yet.")

st.markdown("### Fault Event Log")
if st.session_state.live_events:
    event_df = pd.DataFrame(st.session_state.live_events)
    st.dataframe(event_df, use_container_width=True, hide_index=True)
else:
    st.success("No non-nominal events captured in this run.")
