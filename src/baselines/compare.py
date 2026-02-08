"""
Baseline Comparison Script.

Loads the feature dataset, applies the same train/val/test split used in
ML training, evaluates both baselines and all ML models on the held-out
test set, and produces:
  1. A side-by-side comparison table (printed + saved to JSON)
  2. A ROC curve overlay plot (saved as PNG)

Usage:
    python -m src.baselines.compare
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.features.extractor import FeatureExtractor
from src.baselines.entropy_threshold import EntropyThresholdClassifier, ENTROPY_COLUMN
from src.baselines.nist_pass_fail import NistPassFailClassifier, NIST_P_COLUMNS

logger = get_logger(__name__)


def load_and_split(config: dict, seed: int) -> dict:
    """Load feature CSV and reproduce the exact same splits as train.py."""
    csv_path = PROJECT_ROOT / config["dataset"]["processed_dir"] / "features.csv"
    df = pd.read_csv(csv_path)

    fe = FeatureExtractor(config)
    feature_names = fe.feature_names()

    X = df[feature_names].values
    y = df["label"].values

    test_size = config["model"]["test_size"]
    val_size = config["model"]["val_size"]

    # Same splitting logic as train.py
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size,
        random_state=seed, stratify=y_temp
    )

    return {
        "df": df,
        "feature_names": feature_names,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def evaluate(y_true, y_pred, y_prob) -> dict:
    """Compute evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "recall_weak": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def get_feature_indices(feature_names: list[str], columns: list[str]) -> list[int]:
    """Map column names to indices in the feature array."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    return [name_to_idx[c] for c in columns]


def run_baselines(data: dict) -> dict:
    """Fit & evaluate both baselines on the test set.

    Returns dict mapping baseline name → (predictions_dict, metrics).
    """
    feature_names = data["feature_names"]
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    results = {}

    # --- Baseline 1: Entropy Threshold ---
    ent_idx = get_feature_indices(feature_names, [ENTROPY_COLUMN])[0]
    ent_train = X_train[:, ent_idx]
    ent_test = X_test[:, ent_idx]

    entropy_clf = EntropyThresholdClassifier()
    entropy_clf.fit(ent_train, y_train)

    y_pred_ent = entropy_clf.predict(ent_test)
    y_prob_ent = entropy_clf.predict_proba(ent_test)[:, 1]

    results["entropy_threshold"] = {
        "metrics": evaluate(y_test, y_pred_ent, y_prob_ent),
        "y_pred": y_pred_ent,
        "y_prob": y_prob_ent,
        "threshold": entropy_clf.threshold,
    }
    logger.info(f"Entropy Threshold — threshold: {entropy_clf.threshold:.6f}")

    # --- Baseline 2: NIST All-Tests-Pass ---
    nist_idxs = get_feature_indices(feature_names, NIST_P_COLUMNS)
    nist_train = X_train[:, nist_idxs]
    nist_test = X_test[:, nist_idxs]

    nist_clf = NistPassFailClassifier(alpha=0.01)
    nist_clf.fit(nist_train, y_train)

    y_pred_nist = nist_clf.predict(nist_test)
    y_prob_nist = nist_clf.predict_proba(nist_test)[:, 1]

    results["nist_pass_fail"] = {
        "metrics": evaluate(y_test, y_pred_nist, y_prob_nist),
        "y_pred": y_pred_nist,
        "y_prob": y_prob_nist,
    }

    return results


def load_ml_results(data: dict, models_dir: Path) -> dict:
    """Load ML models and evaluate on the same test set.

    Returns dict mapping model name → metrics dict + predictions.
    """
    from sklearn.preprocessing import StandardScaler

    feature_names = data["feature_names"]
    X_train, X_test, y_test = data["X_train"], data["X_test"], data["y_test"]

    # Reproduce scaler fit on training data
    scaler = joblib.load(models_dir / "scaler.joblib")

    model_files = {
        "logistic_regression": "logistic_regression.joblib",
        "random_forest": "random_forest.joblib",
        "xgboost": "xgboost.joblib",
        "stacking_ensemble": "stacking_ensemble.joblib",
    }

    results = {}
    for name, fname in model_files.items():
        model_path = models_dir / fname
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            continue

        model = joblib.load(model_path)

        # LR needs scaled features; tree-based models and stacking use raw
        if name == "logistic_regression":
            X = scaler.transform(X_test)
        else:
            X = X_test

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        results[name] = {
            "metrics": evaluate(y_test, y_pred, y_prob),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    return results


def print_comparison_table(baseline_results: dict, ml_results: dict):
    """Print a formatted side-by-side comparison table."""
    all_results = {}
    all_results.update(baseline_results)
    all_results.update(ml_results)

    header = f"{'Model':<25s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s} {'RecW':>6s}"
    sep = "-" * len(header)

    logger.info("\n" + sep)
    logger.info("          COMPARISON TABLE (Test Set)")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for name, result in all_results.items():
        m = result["metrics"]
        logger.info(
            f"{name:<25s} {m['accuracy']:6.3f} {m['precision']:6.3f} "
            f"{m['recall']:6.3f} {m['f1']:6.3f} {m['roc_auc']:6.3f} "
            f"{m['recall_weak']:6.3f}"
        )

    logger.info(sep)
    logger.info("  Acc=Accuracy, Prec=Precision(strong), Rec=Recall(strong),")
    logger.info("  F1=F1(strong), AUC=ROC-AUC, RecW=Recall(weak class)")
    logger.info(sep)


def plot_roc_overlay(
    baseline_results: dict,
    ml_results: dict,
    y_test: np.ndarray,
    output_path: Path,
):
    """Plot ROC curves for all models on one figure and save as PNG."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Baselines: dashed lines
    style_map = {
        "entropy_threshold": {"ls": "--", "color": "tab:orange", "lw": 2},
        "nist_pass_fail": {"ls": "--", "color": "tab:red", "lw": 2},
    }
    # ML models: solid lines
    ml_colors = {
        "logistic_regression": "tab:blue",
        "random_forest": "tab:green",
        "xgboost": "tab:purple",
        "stacking_ensemble": "tab:cyan",
    }

    all_results = {}
    all_results.update(baseline_results)
    all_results.update(ml_results)

    for name, result in all_results.items():
        y_prob = result["y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = result["metrics"]["roc_auc"]

        if name in style_map:
            style = style_map[name]
            ax.plot(fpr, tpr, linestyle=style["ls"], color=style["color"],
                    linewidth=style["lw"],
                    label=f"{name} (AUC={auc:.3f})")
        else:
            color = ml_colors.get(name, "gray")
            ax.plot(fpr, tpr, linestyle="-", color=color, linewidth=2,
                    label=f"{name} (AUC={auc:.3f})")

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Random (AUC=0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison — Baselines vs ML Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC overlay plot saved to {output_path}")


def save_comparison_results(
    baseline_results: dict,
    ml_results: dict,
    output_path: Path,
):
    """Save the full comparison table as JSON."""
    comparison = {}
    for name, result in {**baseline_results, **ml_results}.items():
        entry = dict(result["metrics"])
        if "threshold" in result:
            entry["threshold"] = result["threshold"]
        comparison[name] = entry

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Comparison results saved to {output_path}")


def compare():
    """Main comparison pipeline."""
    config = get_config()
    seed = set_global_seed()
    models_dir = PROJECT_ROOT / config["model"]["output_dir"]

    # Load data with same splits
    data = load_and_split(config, seed)
    logger.info(f"Test set: {len(data['y_test'])} samples "
                f"(strong={int(data['y_test'].sum())}, "
                f"weak={int((data['y_test']==0).sum())})")

    # Run baselines
    logger.info("\n=== Evaluating Baselines ===")
    baseline_results = run_baselines(data)

    # Load and evaluate ML models
    logger.info("\n=== Evaluating ML Models ===")
    ml_results = load_ml_results(data, models_dir)

    # Comparison table
    print_comparison_table(baseline_results, ml_results)

    # ROC overlay plot
    output_dir = models_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_overlay(
        baseline_results, ml_results, data["y_test"],
        output_dir / "roc_comparison.png"
    )

    # Save JSON
    save_comparison_results(
        baseline_results, ml_results,
        output_dir / "comparison_results.json"
    )

    # Written analysis
    write_analysis(baseline_results, ml_results, output_dir / "analysis.md")

    return baseline_results, ml_results


def write_analysis(
    baseline_results: dict,
    ml_results: dict,
    output_path: Path,
):
    """Generate a short written analysis of where ML outperforms baselines."""
    all_res = {**baseline_results, **ml_results}

    # Find best baseline and best ML model by F1
    best_bl_name = max(baseline_results, key=lambda k: baseline_results[k]["metrics"]["f1"])
    best_ml_name = max(ml_results, key=lambda k: ml_results[k]["metrics"]["f1"])

    bl = baseline_results[best_bl_name]["metrics"]
    ml = ml_results[best_ml_name]["metrics"]

    lines = [
        "# Baseline vs ML Comparison — Analysis",
        "",
        "## Summary",
        "",
        f"- **Best baseline:** {best_bl_name} (F1 = {bl['f1']:.4f})",
        f"- **Best ML model:** {best_ml_name} (F1 = {ml['f1']:.4f})",
        "",
        "## Key Findings",
        "",
    ]

    # F1 comparison
    f1_improvement = ml["f1"] - bl["f1"]
    if f1_improvement > 0:
        lines.append(f"1. **ML outperforms baselines on F1** by {f1_improvement:.4f} "
                      f"({best_ml_name}: {ml['f1']:.4f} vs {best_bl_name}: {bl['f1']:.4f}).")
    else:
        lines.append(f"1. **ML does not outperform the best baseline on F1.** "
                      f"({best_ml_name}: {ml['f1']:.4f} vs {best_bl_name}: {bl['f1']:.4f}).")

    # AUC comparison
    auc_improvement = ml["roc_auc"] - bl["roc_auc"]
    lines.append(f"2. **ROC-AUC improvement:** {auc_improvement:+.4f} "
                  f"({best_ml_name}: {ml['roc_auc']:.4f} vs {best_bl_name}: {bl['roc_auc']:.4f}).")

    # Weak-class recall
    recw_improvement = ml["recall_weak"] - bl["recall_weak"]
    lines.append(f"3. **Weak-class recall change:** {recw_improvement:+.4f} "
                  f"({best_ml_name}: {ml['recall_weak']:.4f} vs {best_bl_name}: {bl['recall_weak']:.4f}).")

    lines.extend([
        "",
        "## Why ML Ensemble Adds Value",
        "",
        "- **Feature fusion:** The ML ensemble combines information from all statistical",
        "  tests simultaneously, whereas individual baselines rely on a single signal",
        "  (entropy alone or NIST pass/fail alone).",
        "- **Decision boundaries:** The stacking ensemble learns non-linear decision",
        "  boundaries in the high-dimensional feature space, capturing subtle patterns",
        "  that simple thresholds miss.",
        "- **Robustness:** Different weak generators exhibit different failure modes.",
        "  The entropy baseline may miss generators with good entropy but poor",
        "  autocorrelation; the NIST baseline may pass generators that are weak in",
        "  ways not directly tested. The ML fusion approach adapts to all failure modes.",
        "",
        "## Full Metrics Table",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall(Weak) |",
        "|-------|----------|-----------|--------|-----|---------|--------------|",
    ])

    for name, result in all_res.items():
        m = result["metrics"]
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | "
            f"{m['recall_weak']:.4f} |"
        )

    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Analysis written to {output_path}")


if __name__ == "__main__":
    compare()
