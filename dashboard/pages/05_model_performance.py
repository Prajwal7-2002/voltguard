from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dashboard.shared import (
    apply_theme,
    find_metrics_per_class,
    image_if_exists,
    load_model_snapshot,
    metric_value,
    render_page_header,
)
from voltguard.diagnostics.drift import DriftDetector
from voltguard.features.engine import ALL_FEATURES

st.set_page_config(page_title="Model Performance", page_icon="VG", layout="wide")
apply_theme()
render_page_header(
    "Model Performance",
    "Quality checks: aggregate metrics, per-class behavior, and drift baseline compatibility.",
)

try:
    snapshot = load_model_snapshot()
except Exception as exc:
    st.error(f"Unable to load model registry: {exc}")
    st.stop()

version = snapshot["version"]
metrics = snapshot["metrics"]
raw_baseline = snapshot["baseline"] or {}

acc = metric_value(metrics, "test_accuracy", ["accuracy"])
f1 = metric_value(metrics, "test_f1_macro")
prec = metric_value(metrics, "test_precision_macro", ["precision"])
rec = metric_value(metrics, "test_recall_macro", ["recall"])
cv_f1 = metric_value(metrics, "cv_f1_macro_mean")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Active Version", version)
k2.metric("Accuracy", f"{acc:.4f}")
k3.metric("Macro F1", f"{f1:.4f}")
k4.metric("Macro Precision", f"{prec:.4f}")
k5.metric("Macro Recall", f"{rec:.4f}")
if cv_f1:
    st.caption(f"Cross-validation macro F1: {cv_f1:.4f}")

split_type = metrics.get("benchmark_split", "unknown")
split_label = {
    "profile_group_holdout": "Profile-held-out benchmark",
    "random_stratified_rows": "Random row split",
}.get(split_type, str(split_type))

s1, s2, s3, s4 = st.columns(4)
s1.metric("Benchmark Split", split_label)
train_rows_label = (
    f"{int(metrics.get('train_rows', 0)):,}" if metrics.get("train_rows") else "N/A"
)
test_rows_label = (
    f"{int(metrics.get('test_rows', 0)):,}" if metrics.get("test_rows") else "N/A"
)
s2.metric("Train Rows", train_rows_label)
s3.metric("Test Rows", test_rows_label)
if metrics.get("test_groups"):
    s4.metric("Held-Out Profiles", str(metrics["test_groups"]))
else:
    s4.metric("Held-Out Profiles", "N/A")

if split_type != "profile_group_holdout":
    st.warning(
        "This model artifact does not report a profile-held-out benchmark. "
        "Treat accuracy and F1 as optimistic until the model is retrained."
    )

if Path("models/latest").exists() and version != "latest":
    st.caption(
        "Active model is resolved from models/latest.txt. The models/latest directory "
        "is a legacy artifact and is not used by this dashboard."
    )

left, right = st.columns([1.0, 1.0])
with left:
    st.markdown("### Confusion Matrix")
    cm = metrics.get("confusion_matrix")
    cm_labels = metrics.get("confusion_matrix_labels")
    if cm and cm_labels:
        cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Predicted", "y": "Actual", "color": "Count"},
            aspect="auto",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        image_if_exists("results/confusion_matrix.png", f"Confusion matrix for {version}")
with right:
    st.markdown("### SHAP Summary")
    image_if_exists("results/shap_summary.png", f"Global SHAP summary for {version}")

st.markdown("### Per-Class Metrics")
per_class = find_metrics_per_class(metrics)
if per_class:
    rows = []
    for label, data in per_class.items():
        if not isinstance(data, dict):
            continue
        rows.append(
            {
                "fault": str(label),
                "precision": float(data.get("precision", 0.0)),
                "recall": float(data.get("recall", 0.0)),
                "f1": float(data.get("f1-score", data.get("f1", 0.0))),
                "support": int(float(data.get("support", 0))),
            }
        )
    per_class_df = pd.DataFrame(rows).sort_values("f1")
    st.dataframe(per_class_df, use_container_width=True, hide_index=True)
    rare_or_weak = per_class_df[(per_class_df["support"] < 500) | (per_class_df["f1"] < 0.9)]
    if not rare_or_weak.empty:
        st.warning("Rare or weak classes need more scrutiny before production claims.")
        st.dataframe(rare_or_weak, use_container_width=True, hide_index=True)
else:
    st.info("No per-class metric breakdown found in metrics artifact.")

pr_metrics = metrics.get("per_class_precision_recall")
if isinstance(pr_metrics, dict) and pr_metrics:
    st.markdown("### Per-Class Average Precision")
    pr_rows = [
        {
            "fault": label,
            "average_precision": float(values.get("average_precision", 0.0)),
            "pr_curve_points": int(values.get("pr_curve_points", 0)),
        }
        for label, values in pr_metrics.items()
        if isinstance(values, dict)
    ]
    st.dataframe(
        pd.DataFrame(pr_rows).sort_values("average_precision"),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("### Feature Contract Check")
features_used = metrics.get("features_used", [])
if features_used:
    st.write(f"Features declared by model artifact: {len(features_used)}")
    st.code(", ".join(features_used))
else:
    st.warning("No feature list stored in metrics artifact.")

st.markdown("### Drift Baseline Compatibility")
effective_baseline = raw_baseline
repair_note = None
try:
    detector = DriftDetector(version)
    effective_baseline = detector.baseline or {}
    if len(effective_baseline) != len(raw_baseline):
        repair_note = "Runtime baseline was auto-repaired for legacy compatibility."
except Exception as exc:
    st.warning(f"Could not initialize drift detector: {exc}")

missing = [f for f in ALL_FEATURES if f not in effective_baseline]
extra = [f for f in effective_baseline if f not in ALL_FEATURES]

c1, c2, c3 = st.columns(3)
c1.metric("Expected Features", str(len(ALL_FEATURES)))
c2.metric("Baseline Features", str(len(effective_baseline)))
c3.metric("Missing In Baseline", str(len(missing)))

if repair_note:
    st.info(repair_note)

if missing:
    st.error("Baseline is not compatible with current feature schema. Drift score can be invalid.")
    st.write("Missing baseline keys:")
    st.code(", ".join(missing))
if extra:
    st.write("Extra legacy keys present in baseline:")
    st.code(", ".join(extra))

st.markdown("### Drift Probe")
if st.button("Run drift probe on synthetic window", use_container_width=True):
    try:
        detector = DriftDetector(version)
        probe_df = pd.DataFrame([{f: 0.0 for f in ALL_FEATURES} for _ in range(64)])
        result = detector.calculate_drift_score(probe_df)
        st.json(result)
    except Exception as exc:
        st.error(f"Drift probe failed: {exc}")
