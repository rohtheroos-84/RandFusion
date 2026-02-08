"""
Run the full explainability pipeline.

Generates:
  1. Global feature importance (RF + XGBoost)
  2. SHAP analysis with plots
  3. Sample explanations using explain_decision()
  4. Interpretation guide (Markdown)

Usage:
    python -m src.explainability.run_explainability
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.features.extractor import FeatureExtractor
from src.explainability.feature_importance import run_feature_importance
from src.explainability.shap_analysis import run_shap_analysis
from src.explainability.explain import DecisionExplainer, format_explanation

logger = get_logger(__name__)


def generate_sample_explanations(config: dict, output_dir: Path):
    """Generate and save explanations for a few test samples."""
    seed = config.get("random_seed", 42)

    csv_path = PROJECT_ROOT / config["dataset"]["processed_dir"] / "features.csv"
    df = pd.read_csv(csv_path)

    fe = FeatureExtractor(config)
    feature_names = fe.feature_names()

    X = df[feature_names].values
    y = df["label"].values

    # Reproduce splits
    test_size = config["model"]["test_size"]
    val_size = config["model"]["val_size"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, _, y_train, _ = train_test_split(
        X_temp, y_temp, test_size=relative_val_size,
        random_state=seed, stratify=y_temp
    )

    explainer = DecisionExplainer(config)

    # Pick representative samples: a strong, a weak, and a borderline
    strong_idx = [i for i, label in enumerate(y_test) if label == 1]
    weak_idx = [i for i, label in enumerate(y_test) if label == 0]

    examples = []
    explain_dir = output_dir / "sample_explanations"
    explain_dir.mkdir(parents=True, exist_ok=True)

    sample_cases = [
        ("strong_sample", strong_idx[0] if strong_idx else 0),
        ("weak_sample", weak_idx[0] if weak_idx else 0),
        ("strong_sample_2", strong_idx[len(strong_idx) // 2] if len(strong_idx) > 1 else 0),
    ]

    for name, idx in sample_cases:
        exp = explainer.explain_features(X_test[idx], top_k=5)
        formatted = format_explanation(exp)

        logger.info(f"\n--- {name} (true label: {y_test[idx]}) ---")
        logger.info(f"\n{formatted}")

        # Save
        with open(explain_dir / f"{name}.txt", "w", encoding="utf-8") as f:
            f.write(f"True label: {y_test[idx]} ({'STRONG' if y_test[idx] == 1 else 'WEAK'})\n\n")
            f.write(formatted)

        examples.append({
            "name": name,
            "true_label": int(y_test[idx]),
            "predicted_label": exp.predicted_label,
            "confidence": exp.confidence,
            "top_features": [f["feature"] for f in exp.top_positive_features[:3]],
        })

    # Save summary JSON
    with open(explain_dir / "examples_summary.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)

    logger.info(f"\nSample explanations saved to {explain_dir}")


def write_interpretation_guide(output_dir: Path):
    """Write a user-facing interpretation guide."""
    guide = """# RandFusion — Interpretation Guide

## How to Read the Verdict

RandFusion classifies each token batch as **STRONG** or **WEAK**:

- **STRONG (1):** The tokens appear to come from a cryptographically secure
  random number generator (CSPRNG). The statistical patterns are consistent
  with true randomness.
- **WEAK (0):** The tokens show detectable statistical anomalies that suggest
  a flawed or predictable generator.

## Confidence Score

The confidence score (0–100%) indicates how certain the model is.
- **> 90%:** Very confident — the signal is clear.
- **70–90%:** Moderately confident — most features agree.
- **50–70%:** Low confidence — the sample is borderline. Consider
  generating more tokens or running additional tests.

## Understanding SHAP Values

Each prediction is accompanied by SHAP (SHapley Additive exPlanations)
values that show **why** the model made its decision:

- **Positive SHAP value** for a feature → pushes the prediction toward STRONG.
- **Negative SHAP value** → pushes toward WEAK.
- **Larger |SHAP|** → greater influence on the final verdict.

### Key Features to Watch

| Feature | What It Measures | Strong Signal | Weak Signal |
|---------|-----------------|---------------|-------------|
| `shannon_entropy` | Byte-level randomness | Near 8.0 (max for bytes) | Significantly below 8.0 |
| `min_entropy` | Worst-case unpredictability | Close to `shannon_entropy` | Much lower than Shannon |
| `compression_ratio` | Compressibility | ~1.0 (incompressible) | < 1.0 (patterns exist) |
| `nist_frequency_p` | Bit balance | > 0.01 | < 0.01 (biased) |
| `nist_runs_p` | Bit transitions | > 0.01 | < 0.01 (too few or too many runs) |
| `autocorr_lag_*` | Sequential dependency | Near 0.0 | Far from 0.0 (predictable) |
| `run_max` | Longest same-bit streak | Moderate (≤ ~20 for 128K bits) | Very long (> 30) |

### Reading the Plots

1. **SHAP Summary (Bee-swarm):** Each dot is a sample. Red = high feature
   value, blue = low. Dots to the right push toward STRONG; left toward WEAK.
   Features at the top are most important.

2. **SHAP Bar Plot:** Shows mean absolute SHAP value per feature — the
   overall importance ranking.

3. **Dependence Plots:** Show how one feature's value affects the prediction.
   Look for clear trends (e.g., higher entropy → higher SHAP → more likely STRONG).

4. **Waterfall Plots:** Show the step-by-step buildup of a single prediction
   from the base value to the final output. Each bar is one feature's contribution.

## Feature Importance (RF vs XGBoost)

The `feature_importance.png` chart compares how Random Forest and XGBoost
weight features internally. Features that rank highly in both models are
the most reliably informative.

## Common Weak Generator Patterns

| Generator Type | Typical Failure Signals |
|---------------|------------------------|
| **Biased coin** | Low `nist_frequency_p`, low entropy |
| **LCG (small state)** | High autocorrelation, poor runs test |
| **Repeating seed** | Low min-entropy, high compression |
| **XOR collapse** | Multiple NIST test failures, low entropy |
| **Predictable seed** | Subtle autocorrelation patterns |

## Limitations

- The model was trained on synthetic data with specific weak generators.
  Novel weakness patterns not represented in training may not be detected.
- A STRONG classification does not guarantee cryptographic security —
  it means the batch passed the statistical tests the model was trained on.
- Very small batches (< 100 tokens) produce less reliable verdicts due to
  insufficient statistical power.
"""

    guide_path = output_dir / "interpretation_guide.md"
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide)

    logger.info(f"Interpretation guide saved to {guide_path}")


def main():
    """Run the full explainability pipeline."""
    config = get_config()
    set_global_seed()

    models_dir = PROJECT_ROOT / config["model"]["output_dir"]
    output_dir = models_dir / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global feature importance
    logger.info("\n=== Step 1: Global Feature Importance ===")
    importance_df = run_feature_importance(config)

    # 2. SHAP analysis
    logger.info("\n=== Step 2: SHAP Analysis ===")
    shap_values = run_shap_analysis(config)

    # 3. Sample explanations
    logger.info("\n=== Step 3: Sample Explanations ===")
    generate_sample_explanations(config, output_dir)

    # 4. Interpretation guide
    logger.info("\n=== Step 4: Interpretation Guide ===")
    write_interpretation_guide(output_dir)

    logger.info("\n=== Explainability pipeline complete! ===")
    logger.info(f"All outputs saved under {output_dir}/")


if __name__ == "__main__":
    main()
