from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dashboard.shared import apply_theme, load_model_snapshot, metric_value, render_page_header

st.set_page_config(
    page_title="VoltGuard Command Center",
    page_icon="VG",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

render_page_header(
    "VoltGuard Dashboard",
    "Fleet diagnostics, model quality, drift readiness, and explainability status.",
)

try:
    snapshot = load_model_snapshot()
    metrics = snapshot["metrics"]
    version = snapshot["version"]
except Exception as exc:
    st.error(f"Could not load model registry: {exc}")
    st.stop()

accuracy = metric_value(metrics, "test_accuracy", ["accuracy"])
f1_macro = metric_value(metrics, "test_f1_macro")
cv_f1 = metric_value(metrics, "cv_f1_macro_mean")
num_classes = int(metrics.get("num_classes", 5))
rows = int(metrics.get("dataset_rows", 0))
feature_count = len(metrics.get("features_used", []))
benchmark_split = metrics.get("benchmark_split", "unknown")
split_label = {
    "profile_group_holdout": "Profile-held-out",
    "random_stratified_rows": "Random rows",
}.get(benchmark_split, str(benchmark_split))

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Model Version", version)
k2.metric("Test Accuracy", f"{accuracy:.4f}")
k3.metric("Macro F1", f"{f1_macro:.4f}")
k4.metric("CV Macro F1", f"{cv_f1:.4f}" if cv_f1 else "N/A")
k5.metric("Benchmark", split_label)

if benchmark_split != "profile_group_holdout":
    st.warning("Current artifact is not profile-held-out. Treat benchmark scores as optimistic.")

st.markdown("### Registry Snapshot")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Dataset Rows", f"{rows:,}" if rows else "N/A")
with c2:
    st.metric("Features", str(feature_count))
with c3:
    st.metric("Classes", str(num_classes))
with c4:
    test_rows_label = (
        f"{int(metrics.get('test_rows', 0)):,}" if metrics.get("test_rows") else "N/A"
    )
    st.metric("Test Rows", test_rows_label)

left, right = st.columns([1.0, 1.0])
with left:
    st.markdown("### Class Distribution")
    class_distribution = metrics.get("class_distribution", {})
    fault_labels = metrics.get("fault_labels", {})
    if class_distribution:
        dist_rows = []
        for code, count in class_distribution.items():
            dist_rows.append(
                {
                    "fault": fault_labels.get(str(code), str(code)),
                    "rows": int(count),
                }
            )
        dist_df = pd.DataFrame(dist_rows).sort_values("rows", ascending=False)
        fig = px.bar(dist_df, x="fault", y="rows", text_auto=True)
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), xaxis_title="", yaxis_title="Rows")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No class distribution stored in the model metrics.")

with right:
    st.markdown("### Weakest Classes")
    per_class = metrics.get("per_class_report") or metrics.get("per_class") or {}
    if per_class:
        quality_rows = []
        for label, values in per_class.items():
            if isinstance(values, dict):
                quality_rows.append(
                    {
                        "fault": str(label),
                        "precision": float(values.get("precision", 0.0)),
                        "recall": float(values.get("recall", 0.0)),
                        "f1": float(values.get("f1-score", values.get("f1", 0.0))),
                        "support": int(float(values.get("support", 0))),
                    }
                )
        st.dataframe(
            pd.DataFrame(quality_rows).sort_values("f1"),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No per-class metrics stored in the model metrics.")

st.markdown("### Notes")
st.write("- Metrics are read from the active version named in `models/latest.txt`.")
st.write(
    "- Profile-held-out benchmarks are preferred because random row splits can overstate "
    "performance."
)
st.write(
    "- Drift diagnostics are valid only when the baseline feature schema matches the "
    "current feature list."
)
