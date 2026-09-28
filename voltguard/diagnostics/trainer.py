"""Model training pipeline for VoltGuard.

Trains an XGBoost classifier on PMSM telemetry, evaluates it on held-out
drive profiles, and saves versioned model artifacts.

Evaluation design (why the numbers look the way they do):
- Rows are 2 Hz samples, so adjacent rows are near-duplicates. A random row
  split leaks each drive into both train and test and inflates metrics, so
  whole ``profile_id`` drives are held out instead.
- A fault class that occurs in fewer than two drives cannot be both learned
  and tested on an unseen drive. Its drives are pinned to training and the
  class is reported as not evaluable rather than silently scored.
- Classes are heavily imbalanced (~91% Nominal), so training uses balanced
  sample weights and the metrics include an always-Nominal baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
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

MIN_PROFILES_TO_EVALUATE = 2
CV_FOLDS = 5


def _build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
        objective="multi:softprob",
        num_class=len(FAULT_LABELS),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=cfg.model.random_state,
    )


def _pinned_profiles(y: pd.Series, groups: pd.Series) -> tuple[set, list[int]]:
    """Profiles that must stay in training, and the classes that forces out of evaluation."""
    profiles_per_class = groups.groupby(y).nunique()
    rare = [int(c) for c, n in profiles_per_class.items() if n < MIN_PROFILES_TO_EVALUATE]
    pinned = set(groups[y.isin(rare)].unique())
    return pinned, rare


def _profile_folds(groups: pd.Series, pinned: set, n_folds: int, seed: int):
    """Yield (train_idx, test_idx) splitting whole profiles; pinned profiles never go to test."""
    rng = np.random.RandomState(seed)
    candidates = np.array(sorted(set(groups.unique()) - pinned))
    rng.shuffle(candidates)
    for fold in np.array_split(candidates, n_folds):
        test_mask = groups.isin(fold).to_numpy()
        yield np.where(~test_mask)[0], np.where(test_mask)[0]


def _macro_f1(y_true, y_pred, exclude: list[int]) -> float:
    labels = [c for c in np.unique(y_true) if c not in exclude]
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def train_model(
    data_path: str | Path,
    version: str | None = None,
    results_dir: str | Path = "results",
) -> dict:
    """Train fault detection model and save artifacts.

    Args:
        data_path: Path to the PMSM training CSV (must include ``profile_id``).
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
    if "profile_id" not in df.columns:
        raise ValueError("Training data must include profile_id for drive-level holdout.")
    df = engineer_features(df)
    df = generate_fault_codes(df)
    df = df.dropna(subset=ALL_FEATURES).reset_index(drop=True)

    features = df[ALL_FEATURES]
    y = df["fault_code"].astype(int)
    groups = df["profile_id"]

    pinned, rare_classes = _pinned_profiles(y, groups)
    logger.info(
        "Dataset: %d rows, %d profiles, classes=%s. Not evaluable (<%d profiles): %s",
        len(df),
        groups.nunique(),
        dict(y.value_counts().sort_index()),
        MIN_PROFILES_TO_EVALUATE,
        rare_classes,
    )

    # ── Drive-level Train/Test Split ──────────────────────────────────────
    seed = cfg.model.random_state
    n_test_folds = max(2, round(1 / cfg.model.test_split))
    train_idx, test_idx = next(_profile_folds(groups, pinned, n_test_folds, seed))
    x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    # ── Cross-Validation over drives (training portion only) ──────────────
    logger.info("Running %d-fold drive-level cross-validation...", CV_FOLDS)
    cv_scores = []
    for fold_train, fold_test in _profile_folds(
        groups_train.reset_index(drop=True), pinned, CV_FOLDS, seed
    ):
        xs_tr, ys_tr = x_train.iloc[fold_train], y_train.iloc[fold_train]
        xs_te, ys_te = x_train.iloc[fold_test], y_train.iloc[fold_test]
        fold_scaler = StandardScaler().fit(xs_tr)
        fold_model = _build_model()
        fold_model.fit(
            fold_scaler.transform(xs_tr),
            ys_tr,
            sample_weight=compute_sample_weight("balanced", ys_tr),
        )
        preds = fold_model.predict(fold_scaler.transform(xs_te))
        cv_scores.append(_macro_f1(ys_te, preds, rare_classes))
    cv_scores = np.array(cv_scores)
    logger.info("CV F1-macro: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # ── Final Training ────────────────────────────────────────────────────
    logger.info("Training final model on full training set...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = _build_model()
    model.fit(x_train_scaled, y_train, sample_weight=compute_sample_weight("balanced", y_train))

    # ── Evaluation ────────────────────────────────────────────────────────
    logger.info("Evaluating model performance on held-out drives...")
    metrics = evaluate_model(
        model=model,
        x_test_scaled=x_test_scaled,
        y_test=y_test,
        feature_columns=ALL_FEATURES,
        fault_labels=FAULT_LABELS,
        results_dir=results_dir,
    )
    y_pred = model.predict(x_test_scaled)
    always_nominal = np.zeros(len(y_test), dtype=int)
    # Recompute macro scores over evaluable classes only, so all three agree.
    evaluable = [c for c in np.unique(y_test) if c not in rare_classes]
    metrics["test_f1_macro"] = _macro_f1(y_test, y_pred, rare_classes)
    metrics["test_precision_macro"] = float(
        precision_score(y_test, y_pred, labels=evaluable, average="macro", zero_division=0)
    )
    metrics["test_recall_macro"] = float(
        recall_score(y_test, y_pred, labels=evaluable, average="macro", zero_division=0)
    )
    metrics["baseline_always_nominal"] = {
        "test_accuracy": float(accuracy_score(y_test, always_nominal)),
        "test_f1_macro": _macro_f1(y_test, always_nominal, rare_classes),
    }

    metrics.update(
        {
            "cv_f1_macro_mean": float(cv_scores.mean()),
            "cv_f1_macro_std": float(cv_scores.std()),
            "benchmark_split": "profile_group_holdout",
            "cv_split": f"profile_group_{CV_FOLDS}fold",
            "group_column": "profile_id",
            "train_groups": int(groups_train.nunique()),
            "test_groups": int(groups.iloc[test_idx].nunique()),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "class_weighting": "balanced",
            "not_evaluable_classes": {str(c): FAULT_LABELS[c] for c in rare_classes},
            "num_classes": len(FAULT_LABELS),
            "fault_labels": {str(k): v for k, v in FAULT_LABELS.items()},
            "class_distribution": {
                str(k): int(v) for k, v in y.value_counts().sort_index().items()
            },
            "profiles_per_class": {
                str(k): int(v) for k, v in groups.groupby(y).nunique().sort_index().items()
            },
            "dataset_rows": int(len(df)),
            "features_used": ALL_FEATURES,
            "leakage_notes": [
                "Hidden thermal channels (stator_*, pm) define labels and are excluded from features.",
                "Test set is whole held-out drive profiles; CV folds are also drive-level.",
                "Macro F1 excludes classes present in fewer than "
                f"{MIN_PROFILES_TO_EVALUATE} profiles (see not_evaluable_classes).",
            ],
        }
    )

    # ── Save Artifacts ────────────────────────────────────────────────────
    baseline_stats = {
        col: {"mean": float(x_train[col].mean()), "std": float(x_train[col].std())}
        for col in ALL_FEATURES
    }
    saved_version = ModelRegistry().save(
        model, scaler, metrics, version=version, baseline=baseline_stats
    )
    logger.info(
        "Training complete! Model '%s' saved. Held-out F1-macro %.4f (always-Nominal %.4f)",
        saved_version,
        metrics["test_f1_macro"],
        metrics["baseline_always_nominal"]["test_f1_macro"],
    )
    return metrics
