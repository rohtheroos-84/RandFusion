# RandFusion — Interpretation Guide

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
