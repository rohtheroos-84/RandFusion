"""
Final Evaluation & Reporting for RandFusion.

Generates:
  1. Confusion matrices (heatmap) for all models
  2. ROC curves (per-model and overlay)
  3. Precision-Recall curves
  4. Calibration curve (reliability diagram)
  5. Stress tests with edge cases
  6. Final structured report (Markdown)

Usage:
    python -m src.evaluation.evaluate
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.features.extractor import FeatureExtractor

logger = get_logger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Data loading — same splits as train.py
# ───────────────────────────────────────────────────────────────────────────

def load_data_and_models(config: dict) -> dict:
    """Load feature CSV, reproduce splits, and load all models."""
    seed = config.get("random_seed", 42)
    csv_path = PROJECT_ROOT / config["dataset"]["processed_dir"] / "features.csv"
    df = pd.read_csv(csv_path)

    fe = FeatureExtractor(config)
    feature_names = fe.feature_names()

    X = df[feature_names].values
    y = df["label"].values
    generators = df["generator"].values

    test_size = config["model"]["test_size"]
    val_size = config["model"]["val_size"]

    X_temp, X_test, y_temp, y_test, gen_temp, gen_test = train_test_split(
        X, y, generators, test_size=test_size, random_state=seed, stratify=y
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, gen_train, gen_val = train_test_split(
        X_temp, y_temp, gen_temp, test_size=relative_val_size,
        random_state=seed, stratify=y_temp
    )

    models_dir = PROJECT_ROOT / config["model"]["output_dir"]
    scaler = joblib.load(models_dir / "scaler.joblib")

    models = {}
    for name in ["logistic_regression", "random_forest", "xgboost", "stacking_ensemble"]:
        path = models_dir / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)

    return {
        "feature_names": feature_names,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "gen_test": gen_test,
        "scaler": scaler,
        "models": models,
    }


def predict_all(data: dict) -> dict:
    """Run predictions for all models on the test set.

    Returns dict: model_name -> {y_pred, y_prob}.
    """
    results = {}
    for name, model in data["models"].items():
        if name == "logistic_regression":
            X = data["scaler"].transform(data["X_test"])
        else:
            X = data["X_test"]
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        results[name] = {"y_pred": y_pred, "y_prob": y_prob}
    return results


# ───────────────────────────────────────────────────────────────────────────
# 1. Confusion Matrices
# ───────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(predictions: dict, y_test: np.ndarray, output_dir: Path):
    """Plot confusion matrix heatmaps for all models in a 2x2 grid."""
    model_names = list(predictions.keys())
    n = len(model_names)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, name in enumerate(model_names):
        cm = confusion_matrix(y_test, predictions[name]["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Weak", "Strong"],
                    yticklabels=["Weak", "Strong"])
        axes[i].set_xlabel("Predicted", fontsize=11)
        axes[i].set_ylabel("Actual", fontsize=11)
        axes[i].set_title(name.replace("_", " ").title(), fontsize=12)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrices saved")


# ───────────────────────────────────────────────────────────────────────────
# 2. ROC Curves
# ───────────────────────────────────────────────────────────────────────────

def plot_roc_curves(predictions: dict, y_test: np.ndarray, output_dir: Path):
    """Plot ROC curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"logistic_regression": "tab:blue", "random_forest": "tab:green",
              "xgboost": "tab:purple", "stacking_ensemble": "tab:red"}

    for name, pred in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, pred["y_prob"])
        auc = roc_auc_score(y_test, pred["y_prob"])
        color = colors.get(name, "gray")
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models (Test Set)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curves saved")


# ───────────────────────────────────────────────────────────────────────────
# 3. Precision-Recall Curves
# ───────────────────────────────────────────────────────────────────────────

def plot_pr_curves(predictions: dict, y_test: np.ndarray, output_dir: Path):
    """Plot precision-recall curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"logistic_regression": "tab:blue", "random_forest": "tab:green",
              "xgboost": "tab:purple", "stacking_ensemble": "tab:red"}

    for name, pred in predictions.items():
        prec, rec, _ = precision_recall_curve(y_test, pred["y_prob"])
        ap = average_precision_score(y_test, pred["y_prob"])
        color = colors.get(name, "gray")
        ax.plot(rec, prec, color=color, linewidth=2,
                label=f"{name} (AP={ap:.3f})")

    # Baseline: prevalence
    prevalence = y_test.mean()
    ax.axhline(y=prevalence, color="k", linestyle=":", linewidth=1,
               label=f"Baseline (prev={prevalence:.2f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models (Test Set)", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Precision-recall curves saved")


# ───────────────────────────────────────────────────────────────────────────
# 4. Calibration Curve
# ───────────────────────────────────────────────────────────────────────────

def plot_calibration_curves(predictions: dict, y_test: np.ndarray, output_dir: Path):
    """Plot calibration (reliability) diagrams for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"logistic_regression": "tab:blue", "random_forest": "tab:green",
              "xgboost": "tab:purple", "stacking_ensemble": "tab:red"}

    # Left: calibration curve
    ax = axes[0]
    for name, pred in predictions.items():
        prob_true, prob_pred = calibration_curve(
            y_test, pred["y_prob"], n_bins=8, strategy="uniform"
        )
        color = colors.get(name, "gray")
        ax.plot(prob_pred, prob_true, "s-", color=color, linewidth=2,
                markersize=6, label=name)

    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: histogram of predicted probabilities
    ax2 = axes[1]
    for name, pred in predictions.items():
        color = colors.get(name, "gray")
        ax2.hist(pred["y_prob"], bins=20, alpha=0.4, color=color,
                 label=name, edgecolor="white")

    ax2.set_xlabel("Predicted Probability (Strong)", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Predicted Probabilities", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "calibration_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration curves saved")


# ───────────────────────────────────────────────────────────────────────────
# 5. Stress Tests
# ───────────────────────────────────────────────────────────────────────────

def run_stress_tests(config: dict, data: dict, output_dir: Path) -> dict:
    """Run stress tests with edge-case inputs and report results.

    Tests:
      A. Unseen generator: Mersenne Twister (random.Random) as strong-ish
      B. Unseen generator: constant-byte output (worst case weak)
      C. Varying token lengths (64, 256 bits)
      D. Very small batch sizes (10, 50 tokens per batch)
    """
    import random as stdlib_random
    from src.features.extractor import FeatureExtractor

    fe = FeatureExtractor(config)
    feature_names = fe.feature_names()
    models_dir = PROJECT_ROOT / config["model"]["output_dir"]

    # Use Random Forest (best individual model)
    rf_model = data["models"]["random_forest"]
    stacking = data["models"].get("stacking_ensemble")

    results = {}

    # --- A. Unseen generator: Mersenne Twister ---
    logger.info("Stress test A: Mersenne Twister (unseen generator)...")
    batch_size = config["token"]["batch_size"]
    token_bits = config["token"]["length_bits"]

    mt_batch = np.zeros((batch_size, token_bits), dtype=np.uint8)
    mt_rng = stdlib_random.Random(12345)
    for i in range(batch_size):
        raw = bytes([mt_rng.randint(0, 255) for _ in range(token_bits // 8)])
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        mt_batch[i] = bits[:token_bits]

    mt_features = fe.extract(mt_batch)
    mt_vec = np.array([mt_features[n] for n in feature_names]).reshape(1, -1)
    mt_pred = rf_model.predict(mt_vec)[0]
    mt_prob = rf_model.predict_proba(mt_vec)[0]
    results["mersenne_twister"] = {
        "predicted": int(mt_pred),
        "label": "STRONG" if mt_pred == 1 else "WEAK",
        "confidence": float(max(mt_prob)),
        "note": "Mersenne Twister is NOT cryptographically secure but passes many statistical tests.",
    }
    logger.info(f"  MT: predicted={results['mersenne_twister']['label']} "
                f"(conf={results['mersenne_twister']['confidence']:.3f})")

    # --- B. Unseen generator: constant bytes ---
    logger.info("Stress test B: Constant byte output (all 0xAA)...")
    const_batch = np.tile(
        np.unpackbits(np.array([0xAA], dtype=np.uint8)),
        (batch_size, token_bits // 8)
    )[:, :token_bits]

    const_features = fe.extract(const_batch)
    const_vec = np.array([const_features[n] for n in feature_names]).reshape(1, -1)
    const_pred = rf_model.predict(const_vec)[0]
    const_prob = rf_model.predict_proba(const_vec)[0]
    results["constant_bytes"] = {
        "predicted": int(const_pred),
        "label": "STRONG" if const_pred == 1 else "WEAK",
        "confidence": float(max(const_prob)),
        "note": "Constant output has zero entropy. Model should classify as WEAK.",
    }
    logger.info(f"  Constant: predicted={results['constant_bytes']['label']} "
                f"(conf={results['constant_bytes']['confidence']:.3f})")

    # --- C. Varying token lengths ---
    logger.info("Stress test C: Varying token lengths (64, 256 bits)...")
    for alt_bits in [64, 256]:
        # Generate strong batch with different token length
        import secrets
        alt_batch = np.zeros((batch_size, alt_bits), dtype=np.uint8)
        for i in range(batch_size):
            raw = secrets.token_bytes(alt_bits // 8)
            bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
            alt_batch[i] = bits[:alt_bits]

        alt_features = fe.extract(alt_batch)
        alt_vec = np.array([alt_features[n] for n in feature_names]).reshape(1, -1)
        alt_pred = rf_model.predict(alt_vec)[0]
        alt_prob = rf_model.predict_proba(alt_vec)[0]
        results[f"strong_{alt_bits}bit"] = {
            "predicted": int(alt_pred),
            "label": "STRONG" if alt_pred == 1 else "WEAK",
            "confidence": float(max(alt_prob)),
            "note": f"CSPRNG tokens at {alt_bits} bits (trained on 128-bit tokens).",
        }
        logger.info(f"  {alt_bits}-bit strong: predicted={results[f'strong_{alt_bits}bit']['label']} "
                    f"(conf={results[f'strong_{alt_bits}bit']['confidence']:.3f})")

    # --- D. Very small batch sizes ---
    logger.info("Stress test D: Small batch sizes (10, 50 tokens)...")
    for small_n in [10, 50]:
        small_batch = np.zeros((small_n, token_bits), dtype=np.uint8)
        import secrets
        for i in range(small_n):
            raw = secrets.token_bytes(token_bits // 8)
            bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
            small_batch[i] = bits[:token_bits]

        small_features = fe.extract(small_batch)
        small_vec = np.array([small_features[n] for n in feature_names]).reshape(1, -1)
        small_pred = rf_model.predict(small_vec)[0]
        small_prob = rf_model.predict_proba(small_vec)[0]
        results[f"small_batch_{small_n}"] = {
            "predicted": int(small_pred),
            "label": "STRONG" if small_pred == 1 else "WEAK",
            "confidence": float(max(small_prob)),
            "note": f"Strong CSPRNG with only {small_n} tokens (trained on 1000-token batches).",
        }
        logger.info(f"  Batch {small_n}: predicted={results[f'small_batch_{small_n}']['label']} "
                    f"(conf={results[f'small_batch_{small_n}']['confidence']:.3f})")

    # Save results
    with open(output_dir / "stress_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Stress test results saved to {output_dir / 'stress_test_results.json'}")
    return results


# ───────────────────────────────────────────────────────────────────────────
# 6. Per-generator breakdown
# ───────────────────────────────────────────────────────────────────────────

def per_generator_analysis(
    predictions: dict,
    y_test: np.ndarray,
    gen_test: np.ndarray,
    output_dir: Path,
):
    """Analyse accuracy broken down by generator type on the test set."""
    # Use stacking ensemble predictions
    best_name = "stacking_ensemble" if "stacking_ensemble" in predictions else list(predictions.keys())[0]
    y_pred = predictions[best_name]["y_pred"]

    rows = []
    for gen_name in np.unique(gen_test):
        mask = gen_test == gen_name
        n = mask.sum()
        if n == 0:
            continue
        correct = (y_pred[mask] == y_test[mask]).sum()
        acc = correct / n
        rows.append({
            "generator": gen_name,
            "n_samples": int(n),
            "correct": int(correct),
            "accuracy": float(acc),
            "true_label": int(y_test[mask][0]),
        })

    df = pd.DataFrame(rows).sort_values("accuracy")
    df.to_csv(output_dir / "per_generator_accuracy.csv", index=False)

    logger.info("\nPer-generator accuracy (stacking ensemble, test set):")
    for _, row in df.iterrows():
        label = "strong" if row["true_label"] == 1 else "weak"
        logger.info(f"  {row['generator']:25s} ({label:6s}) "
                    f"{row['correct']}/{row['n_samples']} = {row['accuracy']:.3f}")

    return df


# ───────────────────────────────────────────────────────────────────────────
# 7. Final Report
# ───────────────────────────────────────────────────────────────────────────

def generate_final_report(
    predictions: dict,
    y_test: np.ndarray,
    gen_test: np.ndarray,
    stress_results: dict,
    gen_df: pd.DataFrame,
    output_dir: Path,
):
    """Generate a structured Markdown evaluation report."""
    lines = [
        "# RandFusion -- Final Evaluation Report",
        "",
        "## 1. Overview",
        "",
        "RandFusion is an ML-based framework that fuses classical statistical",
        "randomness tests into a single verdict using a stacking ensemble.",
        "",
        f"- **Test set size:** {len(y_test)} samples "
        f"({int(y_test.sum())} strong, {int((y_test == 0).sum())} weak)",
        f"- **Number of features:** 32 (NIST tests + entropy + compression + autocorrelation)",
        f"- **Models evaluated:** {', '.join(predictions.keys())}",
        "",
        "## 2. Test Set Results",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|-------|----------|-----------|--------|-----|---------|",
    ]

    for name, pred in predictions.items():
        y_pred = pred["y_pred"]
        y_prob = pred["y_prob"]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        lines.append(f"| {name} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {auc:.4f} |")

    # Classification report for ensemble
    ensemble_name = "stacking_ensemble" if "stacking_ensemble" in predictions else list(predictions.keys())[-1]
    ensemble_pred = predictions[ensemble_name]["y_pred"]
    report = classification_report(y_test, ensemble_pred, target_names=["Weak (0)", "Strong (1)"])

    lines.extend([
        "",
        f"### Classification Report ({ensemble_name})",
        "",
        "```",
        report,
        "```",
        "",
        "## 3. Per-Generator Accuracy",
        "",
        "| Generator | Type | Samples | Correct | Accuracy |",
        "|-----------|------|---------|---------|----------|",
    ])

    for _, row in gen_df.iterrows():
        gtype = "STRONG" if row["true_label"] == 1 else "WEAK"
        lines.append(
            f"| {row['generator']} | {gtype} | {row['n_samples']} | "
            f"{row['correct']} | {row['accuracy']:.3f} |"
        )

    lines.extend([
        "",
        "## 4. Stress Test Results",
        "",
        "| Test Case | Predicted | Confidence | Notes |",
        "|-----------|-----------|------------|-------|",
    ])

    for case_name, result in stress_results.items():
        lines.append(
            f"| {case_name} | {result['label']} | "
            f"{result['confidence']:.3f} | {result['note']} |"
        )

    lines.extend([
        "",
        "## 5. Calibration Assessment",
        "",
        "See `calibration_curves.png` for the reliability diagram.",
        "A perfectly calibrated model would follow the diagonal: if it predicts",
        "80% probability of STRONG, then 80% of such samples truly are STRONG.",
        "",
        "## 6. Plots Generated",
        "",
        "| Plot | File |",
        "|------|------|",
        "| Confusion Matrices | `confusion_matrices.png` |",
        "| ROC Curves | `roc_curves.png` |",
        "| Precision-Recall Curves | `precision_recall_curves.png` |",
        "| Calibration Curves | `calibration_curves.png` |",
        "",
        "## 7. Findings & Limitations",
        "",
        "### Strengths",
        "- The ML ensemble successfully fuses multiple statistical signals into a",
        "  single verdict, outperforming naive baselines on AUC.",
        "- Random Forest achieves the highest individual F1 score on the test set.",
        "- SHAP explanations provide transparent, feature-level justifications",
        "  for every decision.",
        "",
        "### Limitations",
        "- **Training data scope:** The model was trained on 8 specific weak generators",
        "  and 2 strong generators. Novel weakness patterns may not be detected.",
        "- **Token length sensitivity:** The model was trained on 128-bit tokens.",
        "  Performance on other lengths varies (see stress tests).",
        "- **Batch size dependency:** Very small batches produce less reliable verdicts",
        "  due to insufficient statistical power in the underlying tests.",
        "- **Not a certification tool:** A STRONG classification indicates statistical",
        "  consistency with true randomness, not a cryptographic guarantee.",
        "",
        "### Future Work",
        "- Expand generator diversity (Mersenne Twister variants, real-world tokens)",
        "- Add cross-validation study on generalization to unseen generators",
        "- Build CLI tool for end-user evaluation",
        "- Experiment with neural network classifiers on raw bitstreams",
        "",
        "---",
        "*Report generated by RandFusion evaluation pipeline.*",
        "",
    ])

    report_path = output_dir / "final_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Final report saved to {report_path}")


# ───────────────────────────────────────────────────────────────────────────
# Main pipeline
# ───────────────────────────────────────────────────────────────────────────

def evaluate():
    """Run the complete evaluation pipeline."""
    config = get_config()
    set_global_seed()

    models_dir = PROJECT_ROOT / config["model"]["output_dir"]
    output_dir = models_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load everything
    logger.info("Loading data and models...")
    data = load_data_and_models(config)
    y_test = data["y_test"]

    # Predictions
    logger.info("Running predictions on test set...")
    predictions = predict_all(data)

    # 1. Confusion matrices
    logger.info("\n=== 1. Confusion Matrices ===")
    plot_confusion_matrices(predictions, y_test, output_dir)

    # 2. ROC curves
    logger.info("\n=== 2. ROC Curves ===")
    plot_roc_curves(predictions, y_test, output_dir)

    # 3. Precision-Recall curves
    logger.info("\n=== 3. Precision-Recall Curves ===")
    plot_pr_curves(predictions, y_test, output_dir)

    # 4. Calibration curves
    logger.info("\n=== 4. Calibration Curves ===")
    plot_calibration_curves(predictions, y_test, output_dir)

    # 5. Per-generator analysis
    logger.info("\n=== 5. Per-Generator Analysis ===")
    gen_df = per_generator_analysis(predictions, y_test, data["gen_test"], output_dir)

    # 6. Stress tests
    logger.info("\n=== 6. Stress Tests ===")
    stress_results = run_stress_tests(config, data, output_dir)

    # 7. Final report
    logger.info("\n=== 7. Generating Final Report ===")
    generate_final_report(
        predictions, y_test, data["gen_test"],
        stress_results, gen_df, output_dir
    )

    logger.info("\n=== Evaluation pipeline complete! ===")
    logger.info(f"All outputs saved under {output_dir}/")


if __name__ == "__main__":
    evaluate()
