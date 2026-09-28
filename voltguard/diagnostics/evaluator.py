"""Evaluation module for model performance analysis and plotting."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    fault_labels: dict[int, str],
) -> dict[str, Any]:
    """Build serializable classification metrics for a multiclass fault model."""
    label_ids = sorted(fault_labels)
    label_names = [fault_labels[i] for i in label_ids]
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        digits=3,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    metrics: dict[str, Any] = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision_macro": float(report["macro avg"]["precision"]),
        "test_recall_macro": float(report["macro avg"]["recall"]),
        "test_f1_macro": float(report["macro avg"]["f1-score"]),
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_labels": label_names,
        "per_class_report": {label: report[label] for label in label_names if label in report},
    }

    if y_prob is not None:
        y_bin = label_binarize(y_true, classes=label_ids)
        if y_bin.shape[1] == len(label_ids) and y_prob.shape[1] == len(label_ids):
            per_class_pr = {}
            for idx, label in enumerate(label_names):
                precision, recall, _ = precision_recall_curve(y_bin[:, idx], y_prob[:, idx])
                per_class_pr[label] = {
                    "average_precision": float(
                        average_precision_score(y_bin[:, idx], y_prob[:, idx])
                    ),
                    "pr_curve_points": int(len(precision)),
                }
            metrics["per_class_precision_recall"] = per_class_pr

    return metrics


def evaluate_model(
    model: Any,
    x_test_scaled: np.ndarray,
    y_test: np.ndarray,
    feature_columns: list[str],
    fault_labels: dict[int, str],
    results_dir: Path,
) -> dict[str, Any]:
    """Generates detailed metrics and visualizations for a trained model.

    Args:
        model: Trained model artifact.
        X_test_scaled: Scaled feature arrays for the test set.
        y_test: Ground truth labels.
        feature_columns: List of feature names.
        fault_labels: Mapping of class index to human-readable labels.
        results_dir: Directory path to specify where plots are rendered.

    Returns:
        A dictionary containing computed test metrics.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(x_test_scaled)
    y_prob = model.predict_proba(x_test_scaled) if hasattr(model, "predict_proba") else None

    # Classification Report
    labels = sorted(fault_labels)
    target_names = [fault_labels[i] for i in labels]
    report_str = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=3,
        zero_division=0,
    )
    logger.info("Classification Report:\n%s", report_str)

    # Dictionary of core metrics
    metrics = classification_metrics(y_test, y_pred, y_prob, fault_labels)

    # Confusion Matrix Plot
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=metrics["confusion_matrix_labels"],
        yticklabels=metrics["confusion_matrix_labels"],
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix to %s", cm_path)

    # SHAP Summary Plot
    logger.info("Computing SHAP explanations for summary plot (sampled)...")
    rng = np.random.RandomState(42)
    sample = x_test_scaled[
        rng.choice(len(x_test_scaled), min(2000, len(x_test_scaled)), replace=False)
    ]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):  # older SHAP: one array per class
        shap_values = shap_values[1]
    elif np.ndim(shap_values) == 3:  # newer SHAP: (rows, features, classes)
        shap_values = shap_values[:, :, 1]
    # Class 1 (Stator Overheat) is the most common fault class.
    shap.summary_plot(shap_values, sample, feature_names=feature_columns, show=False)
    shap_path = results_dir / "shap_summary.png"
    plt.savefig(shap_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary to %s", shap_path)

    return metrics
