"""
Training script for the 5-Class Total Powertrain Protection AI.

Trains on 1.3M rows of Kaggle PMSM data using leak-proof features.
Generates: model.pkl, scaler.pkl, metrics.json, confusion_matrix.png, shap_summary.png
"""
# ruff: noqa: I001
import json
import logging
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from voltguard.diagnostics.evaluator import classification_metrics
from voltguard.features.engine import (
    ALL_FEATURES,
    FAULT_LABELS,
    engineer_features,
    generate_fault_codes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

# Paths
DATA_PATH = Path("data/comprehensive_fault_training_data.csv")
MODEL_DIR = Path("models/v1")
RESULTS_DIR = Path("results")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    logging.info("Reading Kaggle PMSM Dataset...")
    if not DATA_PATH.exists():
        logging.error("comprehensive_fault_training_data.csv not found!")
        return

    df = pd.read_csv(DATA_PATH)
    logging.info(f"Loaded {len(df)} rows. Engineering features...")

    # 1. Engineer derived features (thermal_delta, dt_dt, battery_temp, etc.)
    df = engineer_features(df)

    # 2. Generate multi-class fault labels from physical thresholds
    df = generate_fault_codes(df)

    # 3. Drop rows with NaN in engineered columns
    df = df.dropna(subset=ALL_FEATURES)

    # 4. Feature matrix (LEAK-PROOF: no stator_tooth, stator_yoke, pm)
    features = df[ALL_FEATURES]
    y = df["fault_code"]
    groups = df["profile_id"] if "profile_id" in df.columns else None

    # Log class distribution
    class_dist = dict(y.value_counts().sort_index())
    logging.info(f"Dataset: {len(df)} samples, {len(ALL_FEATURES)} features")
    for code, count in class_dist.items():
        pct = count / len(df) * 100
        logging.info(f"  Class {code} ({FAULT_LABELS[code]}): {count} samples ({pct:.1f}%)")

    # 5. Hold out complete drive profiles when profile_id exists.
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(features, y, groups=groups))
        x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        split_metadata = {
            "benchmark_split": "profile_group_holdout",
            "group_column": "profile_id",
            "train_groups": int(groups_train.nunique()),
            "test_groups": int(groups.iloc[test_idx].nunique()),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
        }
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            features, y, test_size=0.2, random_state=42, stratify=y
        )
        groups_train = None
        split_metadata = {
            "benchmark_split": "random_stratified_rows",
            "group_column": None,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        }

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 6. Train XGBoost multi-class classifier
    logging.info("Training XGBoost 5-Class Classifier (leak-proof features)...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=len(FAULT_LABELS),
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
    )
    if groups_train is not None and groups_train.nunique() >= 5:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model,
            x_train_scaled,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="f1_macro",
        )
        split_metadata["cv_split"] = "stratified_profile_group_kfold"
        logging.info(
            "Profile-group CV F1-macro: %.4f +/- %.4f",
            cv_scores.mean(),
            cv_scores.std(),
        )
    else:
        cv_scores = None
        split_metadata["cv_split"] = "not_run"

    model.fit(x_train_scaled, y_train)

    # 7. Evaluate
    y_pred = model.predict(x_test_scaled)
    y_prob = model.predict_proba(x_test_scaled)
    metrics = classification_metrics(y_test, y_pred, y_prob, FAULT_LABELS)
    acc = metrics["test_accuracy"]
    prec = metrics["test_precision_macro"]
    rec = metrics["test_recall_macro"]
    f1 = metrics["test_f1_macro"]

    logging.info(f"Test Accuracy:       {acc:.4f}")
    logging.info(f"Test Precision (M):  {prec:.4f}")
    logging.info(f"Test Recall (M):     {rec:.4f}")
    logging.info(f"Test F1-Macro:       {f1:.4f}")

    logging.info("\n" + classification_report(
        y_test, y_pred,
        target_names=[FAULT_LABELS[i] for i in range(len(FAULT_LABELS))]
    ))

    # 8. Confusion Matrix (5x5)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [FAULT_LABELS[i] for i in range(len(FAULT_LABELS))]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("5-Class Confusion Matrix (Test Set)", fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    logging.info("Saved 5x5 confusion matrix to results/confusion_matrix.png")

    # 9. SHAP Summary
    logging.info("Computing SHAP explanations (sampled for speed)...")
    sample_idx = np.random.RandomState(42).choice(
        len(x_test_scaled),
        size=min(500, len(x_test_scaled)),
        replace=False,
    )
    x_sample = x_test_scaled[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    # For multi-class, shap_values is a list of arrays; use class 1 (Stator) as primary
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], x_sample, feature_names=ALL_FEATURES, show=False)
    else:
        shap.summary_plot(shap_values, x_sample, feature_names=ALL_FEATURES, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=150)
    plt.close()
    logging.info("Saved SHAP summary to results/shap_summary.png")

    # 10. Export model artifacts
    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    metrics.update(split_metadata)
    if cv_scores is not None:
        metrics["cv_f1_macro_mean"] = float(cv_scores.mean())
        metrics["cv_f1_macro_std"] = float(cv_scores.std())
    metrics.update(
        {
            "num_classes": len(FAULT_LABELS),
            "fault_labels": {str(k): v for k, v in FAULT_LABELS.items()},
            "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
            "dataset_rows": int(len(df)),
            "features_used": ALL_FEATURES,
            "leak_proof": True,
            "leakage_notes": [
                "Hidden thermal channels are excluded from model features.",
                "Metrics use held-out drive profiles when profile_id exists.",
            ],
        }
    )

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4, default=str)

    # Also write latest pointer
    with open(Path("models/latest.txt"), "w") as f:
        f.write("v1")

    logging.info("Model artifacts exported to models/v1/. Training complete!")


def main():
    train_model()


if __name__ == "__main__":
    main()
