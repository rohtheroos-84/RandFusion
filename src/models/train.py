"""
ML model training pipeline for RandFusion.

Loads the feature matrix, splits into train/val/test, trains three base
models (Logistic Regression, Random Forest, XGBoost), builds a stacking
ensemble, evaluates everything, and saves trained models.

Usage:
    python -m src.models.train
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.features.extractor import FeatureExtractor

logger = get_logger(__name__)


def load_feature_data(config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Load the feature CSV and return DataFrame + feature column names."""
    csv_path = PROJECT_ROOT / config["dataset"]["processed_dir"] / "features.csv"
    df = pd.read_csv(csv_path)

    fe = FeatureExtractor(config)
    feature_names = fe.feature_names()

    logger.info(f"Loaded {len(df)} samples with {len(feature_names)} features")
    return df, feature_names


def split_data(
    df: pd.DataFrame,
    feature_names: list[str],
    test_size: float,
    val_size: float,
    seed: int,
) -> dict:
    """Split data into train/validation/test sets (stratified).

    Returns a dict with X_train, X_val, X_test, y_train, y_val, y_test,
    and the generator names for each split.
    """
    X = df[feature_names].values
    y = df["label"].values
    generators = df["generator"].values

    # First split: separate test set
    X_temp, X_test, y_temp, y_test, gen_temp, gen_test = train_test_split(
        X, y, generators, test_size=test_size, random_state=seed, stratify=y
    )

    # Second split: separate validation from training
    # Adjust val_size relative to remaining data
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, gen_train, gen_val = train_test_split(
        X_temp, y_temp, gen_temp, test_size=relative_val_size,
        random_state=seed, stratify=y_temp
    )

    logger.info(f"Split: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    logger.info(f"  Train — strong: {(y_train==1).sum()}, weak: {(y_train==0).sum()}")
    logger.info(f"  Val   — strong: {(y_val==1).sum()}, weak: {(y_val==0).sum()}")
    logger.info(f"  Test  — strong: {(y_test==1).sum()}, weak: {(y_test==0).sum()}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "gen_train": gen_train, "gen_val": gen_val, "gen_test": gen_test,
    }


def train_base_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: StandardScaler,
    seed: int,
) -> dict:
    """Train the three base models and evaluate on validation set.

    Returns dict mapping model name → (model, val_metrics).
    """
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=seed, C=1.0, solver="lbfgs",
            class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=seed, n_jobs=-1, class_weight="balanced"
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=seed, eval_metric="logloss",
            use_label_encoder=False, verbosity=0,
            scale_pos_weight=1.0
        ),
    }

    results = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")

        # LR uses scaled data; tree models use raw
        if name == "logistic_regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            y_prob = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

        metrics = evaluate_predictions(y_val, y_pred, y_prob)
        results[name] = {"model": model, "metrics": metrics}

        logger.info(f"  {name} — Acc: {metrics['accuracy']:.4f}, "
                     f"F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")

    return results


def build_stacking_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    cv_folds: int,
) -> dict:
    """Build a stacking ensemble with Logistic Regression as meta-classifier.

    The stacking classifier uses cross-validated predictions from base models
    as input to the meta-classifier, avoiding data leakage.

    Note: StackingClassifier handles scaling internally for LR via pipeline,
    but tree models don't need scaling. We pass raw features and let
    sklearn handle the stacking logic.
    """
    logger.info("Building stacking ensemble...")

    # Wrap LR in a Pipeline with scaler to prevent convergence issues
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000, random_state=seed, C=1.0, class_weight="balanced"
        )),
    ])

    base_estimators = [
        ("lr", lr_pipeline),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=seed, n_jobs=-1, class_weight="balanced"
        )),
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=seed, eval_metric="logloss",
            use_label_encoder=False, verbosity=0,
            scale_pos_weight=1.0
        )),
    ]

    # Meta-classifier: Logistic Regression (keeps things interpretable)
    meta_clf = LogisticRegression(max_iter=2000, random_state=seed)

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_clf,
        cv=cv_folds,
        stack_method="predict_proba",
        passthrough=False,  # Only use base model predictions as meta-features
        n_jobs=-1,
    )

    stacking.fit(X_train, y_train)

    # Evaluate on validation
    y_pred = stacking.predict(X_val)
    y_prob = stacking.predict_proba(X_val)[:, 1]
    metrics = evaluate_predictions(y_val, y_pred, y_prob)

    logger.info(f"  Stacking — Acc: {metrics['accuracy']:.4f}, "
                 f"F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")

    return {"model": stacking, "metrics": metrics}


def evaluate_predictions(y_true, y_pred, y_prob) -> dict:
    """Compute all evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_on_test(
    model, X_test, y_test, scaler=None, needs_scaling=False
) -> dict:
    """Final evaluation on the held-out test set."""
    X = scaler.transform(X_test) if needs_scaling else X_test
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return evaluate_predictions(y_test, y_pred, y_prob)


def save_artifacts(
    models_dir: Path,
    scaler: StandardScaler,
    base_results: dict,
    ensemble_result: dict,
    test_results: dict,
    feature_names: list[str],
    split_info: dict,
):
    """Save trained models, scaler, and results to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save scaler
    joblib.dump(scaler, models_dir / "scaler.joblib")

    # Save base models
    for name, result in base_results.items():
        joblib.dump(result["model"], models_dir / f"{name}.joblib")

    # Save ensemble
    joblib.dump(ensemble_result["model"], models_dir / "stacking_ensemble.joblib")

    # Save results summary
    results_summary = {
        "feature_names": feature_names,
        "split_sizes": {
            "train": int(len(split_info["y_train"])),
            "val": int(len(split_info["y_val"])),
            "test": int(len(split_info["y_test"])),
        },
        "validation_results": {
            name: result["metrics"] for name, result in base_results.items()
        },
        "stacking_validation": ensemble_result["metrics"],
        "test_results": test_results,
    }

    with open(models_dir / "results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"All artifacts saved to {models_dir}/")


def train():
    """Full training pipeline."""
    config = get_config()
    seed = set_global_seed()

    model_cfg = config["model"]
    models_dir = PROJECT_ROOT / model_cfg["output_dir"]

    # Load data
    df, feature_names = load_feature_data(config)

    # Split
    split = split_data(
        df, feature_names,
        test_size=model_cfg["test_size"],
        val_size=model_cfg["val_size"],
        seed=seed,
    )

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(split["X_train"])

    # Train base models (evaluated on validation set)
    logger.info("\n=== Training Base Models ===")
    base_results = train_base_models(
        split["X_train"], split["y_train"],
        split["X_val"], split["y_val"],
        scaler, seed
    )

    # Build stacking ensemble
    logger.info("\n=== Building Stacking Ensemble ===")
    ensemble_result = build_stacking_ensemble(
        split["X_train"], split["y_train"],
        split["X_val"], split["y_val"],
        seed, model_cfg["cv_folds"]
    )

    # Final evaluation on held-out test set
    logger.info("\n=== Final Test Set Evaluation ===")
    test_results = {}

    for name, result in base_results.items():
        needs_scaling = (name == "logistic_regression")
        metrics = evaluate_on_test(
            result["model"], split["X_test"], split["y_test"],
            scaler=scaler, needs_scaling=needs_scaling
        )
        test_results[name] = metrics
        logger.info(f"  {name:25s} — Acc: {metrics['accuracy']:.4f}, "
                     f"F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")

    # Ensemble on test set
    ensemble_test = evaluate_on_test(
        ensemble_result["model"], split["X_test"], split["y_test"]
    )
    test_results["stacking_ensemble"] = ensemble_test
    logger.info(f"  {'stacking_ensemble':25s} — Acc: {ensemble_test['accuracy']:.4f}, "
                 f"F1: {ensemble_test['f1']:.4f}, AUC: {ensemble_test['roc_auc']:.4f}")

    # Print classification report for ensemble
    y_pred = ensemble_result["model"].predict(split["X_test"])
    logger.info("\n=== Ensemble Classification Report (Test Set) ===")
    report = classification_report(
        split["y_test"], y_pred,
        target_names=["Weak (0)", "Strong (1)"]
    )
    logger.info(f"\n{report}")

    # Save everything
    save_artifacts(
        models_dir, scaler, base_results, ensemble_result,
        test_results, feature_names, split
    )

    return test_results


if __name__ == "__main__":
    train()
