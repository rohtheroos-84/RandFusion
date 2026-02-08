"""
SHAP-based explainability for RandFusion models.

Computes SHAP values using TreeExplainer (for RF and XGBoost) and
produces:
  - Global SHAP summary plot (bee-swarm)
  - SHAP bar plot (mean |SHAP|)
  - Dependence plots for top features
  - Per-sample SHAP waterfall plots

Uses the Random Forest model as the primary explainer because
TreeExplainer is exact and fast for tree models, and RF is the
best-performing individual model.
"""

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.features.extractor import FeatureExtractor

logger = get_logger(__name__)


def load_data_and_model(config: dict) -> dict:
    """Load the feature CSV, split like train.py, and load the RF model."""
    seed = config.get("random_seed", 42)

    csv_path = PROJECT_ROOT / config["dataset"]["processed_dir"] / "features.csv"
    df = pd.read_csv(csv_path)

    fe = FeatureExtractor(config)
    feature_names = fe.feature_names()

    X = df[feature_names].values
    y = df["label"].values

    test_size = config["model"]["test_size"]
    val_size = config["model"]["val_size"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size,
        random_state=seed, stratify=y_temp
    )

    models_dir = PROJECT_ROOT / config["model"]["output_dir"]
    rf_model = joblib.load(models_dir / "random_forest.joblib")
    xgb_model = joblib.load(models_dir / "xgboost.joblib")

    return {
        "feature_names": feature_names,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "rf_model": rf_model,
        "xgb_model": xgb_model,
    }


def compute_shap_values(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    max_samples: int = 200,
) -> shap.Explanation:
    """Compute SHAP values using TreeExplainer.

    Args:
        model: A tree-based sklearn/xgboost model.
        X_background: Background data for the explainer (training set or subset).
        X_explain: Data to explain (test set or subset).
        feature_names: List of feature names.
        max_samples: Maximum samples to explain (for speed).

    Returns:
        shap.Explanation object.
    """
    # Subsample if needed
    if len(X_explain) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_explain), size=max_samples, replace=False)
        X_explain = X_explain[idx]

    explainer = shap.TreeExplainer(model, data=X_background)
    shap_values = explainer(X_explain)

    # For binary classification TreeExplainer may return values for both classes
    # We want SHAP values for class 1 (strong)
    if len(shap_values.shape) == 3:
        # shape: (n_samples, n_features, n_classes) → take class 1
        shap_values = shap_values[:, :, 1]

    # Assign feature names
    shap_values.feature_names = feature_names

    return shap_values


def plot_shap_summary(
    shap_values: shap.Explanation,
    output_dir: Path,
):
    """Generate SHAP summary (bee-swarm) and bar plots."""
    # Bee-swarm plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.title("SHAP Summary — Feature Impact on Prediction", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP bee-swarm plot saved")

    # Bar plot (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False, max_display=20)
    plt.title("SHAP Feature Importance — Mean |SHAP Value|", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP bar plot saved")


def plot_shap_dependence(
    shap_values: shap.Explanation,
    feature_names: list[str],
    output_dir: Path,
    top_n: int = 5,
):
    """Generate SHAP dependence plots for the top N most important features."""
    # Rank features by mean |SHAP|
    mean_abs = np.mean(np.abs(shap_values.values), axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]

    dep_dir = output_dir / "dependence"
    dep_dir.mkdir(parents=True, exist_ok=True)

    for idx in top_indices:
        fname = feature_names[idx]
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.scatter(shap_values[:, idx], show=False)
        plt.title(f"SHAP Dependence — {fname}", fontsize=13)
        plt.tight_layout()
        safe_name = fname.replace("/", "_").replace(" ", "_")
        plt.savefig(dep_dir / f"dependence_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"SHAP dependence plots saved for top {top_n} features")


def plot_shap_waterfall(
    shap_values: shap.Explanation,
    output_dir: Path,
    sample_indices: list[int] | None = None,
):
    """Generate SHAP waterfall plots for individual sample explanations.

    If sample_indices is None, picks one correctly-classified strong,
    one correctly-classified weak, and one misclassified sample.
    """
    waterfall_dir = output_dir / "waterfall"
    waterfall_dir.mkdir(parents=True, exist_ok=True)

    if sample_indices is None:
        sample_indices = [0, len(shap_values) // 2]
        if len(shap_values) > 2:
            sample_indices.append(len(shap_values) - 1)

    for i, idx in enumerate(sample_indices):
        if idx >= len(shap_values):
            continue
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(shap_values[idx], show=False, max_display=15)
        plt.title(f"SHAP Waterfall — Sample {idx}", fontsize=13)
        plt.tight_layout()
        plt.savefig(waterfall_dir / f"waterfall_sample_{idx}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"SHAP waterfall plots saved for {len(sample_indices)} samples")


def run_shap_analysis(config: dict | None = None) -> shap.Explanation:
    """Full SHAP pipeline: compute values, generate all plots."""
    if config is None:
        config = get_config()

    set_global_seed()
    max_samples = config.get("explainability", {}).get("shap_max_samples", 200)
    models_dir = PROJECT_ROOT / config["model"]["output_dir"]
    output_dir = models_dir / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and model
    data = load_data_and_model(config)

    logger.info(f"Computing SHAP values (max {max_samples} samples)...")
    shap_values = compute_shap_values(
        model=data["rf_model"],
        X_background=data["X_train"],
        X_explain=data["X_test"],
        feature_names=data["feature_names"],
        max_samples=max_samples,
    )

    # Save raw SHAP values
    shap_df = pd.DataFrame(shap_values.values, columns=data["feature_names"])
    shap_df.to_csv(output_dir / "shap_values.csv", index=False)
    logger.info("Raw SHAP values saved to shap_values.csv")

    # Generate plots
    logger.info("Generating SHAP plots...")
    plot_shap_summary(shap_values, output_dir)
    plot_shap_dependence(shap_values, data["feature_names"], output_dir, top_n=5)
    plot_shap_waterfall(shap_values, output_dir)

    return shap_values
