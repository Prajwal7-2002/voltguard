"""Model training pipeline for VoltGuard.

Trains an XGBoost classifier on CAN bus sensor data, evaluates performance,
generates SHAP explanations, and saves versioned model artifacts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import cfg
from voltguard.core.logging import get_logger
from voltguard.diagnostics.evaluator import evaluate_model
from voltguard.diagnostics.registry import ModelRegistry
from voltguard.features.engine import (
    ALL_FEATURES,
    FAULT_LABELS,
    engineer_features,
    generate_fault_codes,
)

logger = get_logger(__name__)


def train_model(
    data_path: str | Path,
    version: str | None = None,
    results_dir: str | Path = "results",
) -> dict:
    """Train fault detection model and save artifacts.

    Args:
        data_path: Path to the training CSV file.
        version: Specific version string (e.g. 'v2'). If None, auto-increments.
        results_dir: Directory to save evaluation plots.

    Returns:
        Dictionary of evaluation metrics.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & Engineer Features ──────────────────────────────────────────
    logger.info("Loading training data from %s", data_path)
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    df = generate_fault_codes(df)
    df = df.dropna(subset=ALL_FEATURES)

    features = df[ALL_FEATURES]
    y = df["fault_code"]
    groups = df["profile_id"] if "profile_id" in df.columns else None

    logger.info(
        "Dataset: %d samples, %d features, classes=%s",
        len(df),
        len(ALL_FEATURES),
        dict(y.value_counts()),
    )

    # ── Stratified Train/Test Split ───────────────────────────────────────
    if groups is not None:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.model.test_split,
            random_state=cfg.model.random_state,
        )
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
            features,
            y,
            test_size=cfg.model.test_split,
            stratify=y,
            random_state=cfg.model.random_state,
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

    # ── Cross-Validation ──────────────────────────────────────────────────
    logger.info("Running 5-fold stratified cross-validation...")
    model = XGBClassifier(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
        objective="multi:softprob",
        num_class=len(FAULT_LABELS),
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=cfg.model.random_state,
    )

    if groups_train is not None and groups_train.nunique() >= 5:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.model.random_state)
        cv_scores = cross_val_score(
            model,
            x_train_scaled,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="f1_macro",
        )
        split_metadata["cv_split"] = "stratified_profile_group_kfold"
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.model.random_state)
        cv_scores = cross_val_score(model, x_train_scaled, y_train, cv=cv, scoring="f1_macro")
        split_metadata["cv_split"] = "stratified_row_kfold"
    logger.info("CV F1-macro: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # ── Final Training ────────────────────────────────────────────────────
    logger.info("Training final model on full training set...")
    model.fit(x_train_scaled, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    logger.info("Evaluating model performance...")
    metrics = evaluate_model(
        model=model,
        x_test_scaled=x_test_scaled,
        y_test=y_test,
        feature_columns=ALL_FEATURES,
        fault_labels=FAULT_LABELS,
        results_dir=results_dir,
    )

    # Append cross-val metrics
    metrics["cv_f1_macro_mean"] = float(cv_scores.mean())
    metrics["cv_f1_macro_std"] = float(cv_scores.std())
    metrics.update(split_metadata)
    metrics["num_classes"] = len(FAULT_LABELS)
    metrics["fault_labels"] = {str(k): v for k, v in FAULT_LABELS.items()}
    metrics["class_distribution"] = {
        str(k): int(v) for k, v in y.value_counts().sort_index().items()
    }
    metrics["dataset_rows"] = int(len(df))
    metrics["features_used"] = ALL_FEATURES
    metrics["leakage_notes"] = [
        "Hidden thermal channels are excluded from model features.",
        "When profile_id exists, metrics use held-out drive profiles "
        "instead of random row holdout.",
    ]

    # ── Save Artifacts ────────────────────────────────────────────────────
    logger.info("Saving model to registry...")

    # Generate baseline statistics for drift detection
    baseline_stats = {}
    for col in ALL_FEATURES:
        baseline_stats[col] = {
            "mean": float(x_train[col].mean()),
            "std": float(x_train[col].std()),
        }

    registry = ModelRegistry()
    saved_version = registry.save(model, scaler, metrics, version=version, baseline=baseline_stats)
    logger.info(
        "Training complete! Model version '%s' saved. Test F1-macro: %.4f",
        saved_version,
        metrics.get("test_f1_macro", 0),
    )

    return metrics
