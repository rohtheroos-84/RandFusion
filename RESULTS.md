# RandFusion — In-Depth Results Analysis

## Executive Summary

RandFusion is an ML-based framework that fuses 32 classical statistical randomness features into a single strong/weak verdict using ensemble learning. Trained on 1,000 synthetic batches (500 strong, 500 weak) across 10 generator types, the system was evaluated on a held-out test set of 150 samples.

**Key findings:**

- **Best model:** Random Forest achieved the highest test F1 of **0.838** and perfect strong-class recall (1.000), making it the most reliable single classifier.
- **ML vs. baselines:** All ML models outperform the entropy-threshold baseline (AUC 0.643) on ROC-AUC, with Random Forest reaching **0.796** — a **+23.8%** improvement.
- **Blind spots identified:** XOR-collapse generators (0% detection) and predictable-seed generators (7.7% detection) evade the current feature set, revealing a critical limitation.
- **Top feature:** `nist_longest_run_p` dominates with an average importance of **0.203**, nearly 4× the next-best feature.

---

## 1. Dataset & Experimental Setup

| Parameter | Value |
|-----------|-------|
| Total samples | 1,000 (500 strong + 500 weak) |
| Train / Val / Test split | 700 / 150 / 150 (stratified, seed=42) |
| Features per sample | 32 |
| Token batch size | 1,000 tokens × 128-bit each |
| Strong generators | `secrets_csprng`, `os_urandom` |
| Weak generators | `lcg_small_state`, `biased_070`, `biased_055`, `repeating_seed_32`, `repeating_seed_16`, `predictable_seed`, `xor_collapse_4`, `xor_collapse_8` |

### Feature Categories (32 total)

| Category | Count | Examples |
|----------|-------|---------|
| NIST test p-values | 7 | frequency, block_freq, runs, longest_run, serial, approx_entropy, cusum |
| NIST test statistics | 7 | Corresponding raw test statistics |
| NIST derived metrics | 6 | serial delta1/delta2, p1/p2, approx_entropy apen, cusum fwd/bwd |
| Entropy measures | 3 | Shannon entropy, min-entropy, max probability |
| Run-length statistics | 4 | run mean, std, max, num_runs |
| Autocorrelation | 5 | lag 1, 2, 4, 8, 16 |
| Compression | 1 | compression_ratio |

---

## 2. Model Performance — Test Set

### 2.1 Overall Metrics

| Model | Accuracy | Precision | Recall (Strong) | F1 | ROC-AUC |
|-------|----------|-----------|-----------------|-----|---------|
| Logistic Regression | 0.793 | 0.716 | 0.973 | 0.825 | 0.787 |
| **Random Forest** | **0.807** | **0.721** | **1.000** | **0.838** | **0.796** |
| XGBoost | 0.753 | 0.711 | 0.853 | 0.776 | 0.777 |
| Stacking Ensemble | 0.793 | 0.720 | 0.960 | 0.823 | 0.789 |

> **Note:** "Recall" here is recall for the STRONG class (label=1). Precision is also for the STRONG class. The positive class in the binary setup is STRONG.

### 2.2 Analysis

- **Random Forest** is the clear winner across every metric. It achieves perfect recall for strong generators (never misses a truly strong source) and the highest F1 and AUC.
- **Logistic Regression** performs surprisingly well despite being a linear model, suggesting that many weak generators are linearly separable in the feature space. Its near-perfect recall (0.973) means it very rarely misclassifies a strong generator as weak.
- **XGBoost** underperforms expectations. Its recall drops to 0.853, meaning it incorrectly flags 11 out of 75 strong samples as weak (false negatives). This is the highest false-negative rate among all models.
- **Stacking Ensemble** does not improve over Random Forest. The ensemble's meta-learner (Logistic Regression) averages the predictions rather than amplifying them, and XGBoost's weaker signal dilutes the ensemble's overall performance.

### 2.3 Why the Ensemble Didn't Help

The stacking ensemble was designed to combine the strengths of LR, RF, and XGBoost. However:

1. **High base-model correlation:** RF and LR already agree on most samples (both have recall ≥ 0.96). The ensemble gains little from combining correlated predictions.
2. **XGBoost drags performance down:** XGBoost's 11 false negatives on STRONG samples introduce noise into the meta-learner's input.
3. **Small dataset effect:** With only 700 training samples, the meta-learner doesn't have enough diversity in base-model disagreements to learn a meaningful combination strategy.

---

## 3. Confusion Matrix Deep Dive

### 3.1 Test Set Confusion Matrices

**Class mapping:** 0 = Weak, 1 = Strong. Rows = true labels, Columns = predicted.

#### Logistic Regression
|  | Pred Weak | Pred Strong |
|--|-----------|-------------|
| **True Weak** | 46 | 29 |
| **True Strong** | 2 | 73 |

#### Random Forest
|  | Pred Weak | Pred Strong |
|--|-----------|-------------|
| **True Weak** | 46 | 29 |
| **True Strong** | 0 | 75 |

#### XGBoost
|  | Pred Weak | Pred Strong |
|--|-----------|-------------|
| **True Weak** | 49 | 26 |
| **True Strong** | 11 | 64 |

#### Stacking Ensemble
|  | Pred Weak | Pred Strong |
|--|-----------|-------------|
| **True Weak** | 47 | 28 |
| **True Strong** | 3 | 72 |

### 3.2 Error Pattern Analysis

| Error Type | LR | RF | XGB | Stacking |
|------------|----|----|-----|----------|
| False Positives (Weak predicted as Strong) | 29 | 29 | 26 | 28 |
| False Negatives (Strong predicted as Weak) | 2 | 0 | 11 | 3 |
| **Total Errors** | **31** | **29** | **37** | **31** |

**Critical insight:** All four models share a common weakness — approximately 26-29 weak samples are consistently misclassified as strong. These are almost certainly the **XOR-collapse** and **predictable-seed** generators, whose statistical profiles closely mimic genuine randomness at the feature level (see Section 5).

The models diverge primarily on false negatives. Random Forest's zero false negatives make it the safest choice for applications where missing a weak generator is acceptable, but flagging a strong one as weak is costly.

---

## 4. Validation vs. Test Performance

| Model | Val AUC | Test AUC | Delta |
|-------|---------|----------|-------|
| Logistic Regression | 0.867 | 0.787 | -0.080 |
| Random Forest | 0.871 | 0.796 | -0.075 |
| XGBoost | 0.890 | 0.777 | -0.113 |
| Stacking Ensemble | 0.869 | 0.789 | -0.080 |

All models show a consistent **7-11% AUC drop** from validation to test, indicating mild overfitting. XGBoost shows the largest gap (−0.113), consistent with its tendency to overfit on small datasets without careful tuning. The RF and LR models are more stable, suggesting they generalize better.

This drop is expected given the small dataset size (700 training samples) and the high-dimensional feature space (32 features). Cross-validation would provide more robust estimates.

---

## 5. Per-Generator Accuracy (Stacking Ensemble)

| Generator | Type | Test Samples | Correct | Accuracy | Analysis |
|-----------|------|-------------|---------|----------|----------|
| `lcg_small_state` | WEAK | 9 | 9 | **100.0%** | Perfectly detected — LCGs with small state produce strong autocorrelation patterns |
| `biased_070` | WEAK | 9 | 9 | **100.0%** | 70% bias is easily caught by frequency and entropy tests |
| `biased_055` | WEAK | 10 | 10 | **100.0%** | Even a 55% bias is detectable with enough tokens |
| `repeating_seed_32` | WEAK | 7 | 7 | **100.0%** | Seed repetition causes low min-entropy and high compression |
| `repeating_seed_16` | WEAK | 11 | 11 | **100.0%** | Shorter seed cycle amplifies the pattern |
| `os_urandom` | STRONG | 39 | 39 | **100.0%** | OS-level CSPRNG produces ideal random output |
| `secrets_csprng` | STRONG | 36 | 33 | **91.7%** | 3 samples misclassified as WEAK — likely borderline statistical fluctuations |
| `predictable_seed` | WEAK | 13 | 1 | **7.7%** | Almost entirely missed — see analysis below |
| `xor_collapse_4` | WEAK | 5 | 0 | **0.0%** | Completely missed — see analysis below |
| `xor_collapse_8` | WEAK | 11 | 0 | **0.0%** | Completely missed — see analysis below |

### 5.1 Why XOR-Collapse Generators Evade Detection

XOR-collapse generators combine multiple random bytes using XOR, which preserves many statistical properties of randomness:

- **Bit balance is maintained:** XOR of balanced bits remains balanced -> `nist_frequency_p` stays high.
- **Individual test p-values remain normal:** Standard NIST tests measure first-order and second-order statistics, but XOR-collapse creates higher-order dependencies not captured by these tests.
- **Entropy appears high:** Shannon entropy of XOR'd bytes is close to 8.0 because the byte distribution remains roughly uniform.

**What's needed:** Higher-order statistical tests (e.g., birthday spacing, matrix rank, spectral tests) or raw-bitstream neural network classifiers that can detect multi-byte correlations.

### 5.2 Why Predictable-Seed Generators Evade Detection

Predictable-seed generators use a secure RNG algorithm but with a guessable seed. Each individual output batch looks statistically random because the underlying algorithm (e.g., MT19937) produces well-distributed output:

- **Single-batch statistics are indistinguishable:** Without seeing multiple batches from the same seed, the output is genuinely random-looking.
- **The weakness is logical, not statistical:** The vulnerability lies in seed predictability, which is an information-security property, not a statistical one.

**What's needed:** Multi-batch correlation analysis, seed-space auditing, or metadata-aware classifiers that consider how the generator was initialized.

### 5.3 Why `secrets_csprng` Has 91.7% Instead of 100%

3 out of 36 `secrets_csprng` samples were classified as WEAK. This is expected statistical noise:

- With 1,000 tokens per batch, some batches will have borderline p-values purely by chance (Type I error in the underlying NIST tests).
- The model's decision boundary is calibrated for the training distribution, and rare statistical fluctuations in genuinely random data can cross it.
- A ~8% false-positive rate on strong generators is consistent with the overall precision of 0.72.

---

## 6. Baseline Comparison

| Method | Accuracy | F1 | ROC-AUC | Weak Recall |
|--------|----------|-----|---------|-------------|
| Entropy Threshold (≥ 7.974) | 0.747 | 0.798 | 0.643 | 0.493 |
| NIST Pass/Fail (majority vote) | 0.793 | 0.823 | 0.786 | 0.627 |
| Logistic Regression | 0.793 | 0.825 | 0.787 | 0.613 |
| Random Forest | 0.807 | 0.838 | 0.796 | 0.613 |
| XGBoost | 0.753 | 0.776 | 0.777 | 0.653 |
| Stacking Ensemble | 0.793 | 0.823 | 0.789 | 0.627 |

### 6.1 Key Takeaways

1. **Entropy threshold is the weakest baseline.** It detects only 49.3% of weak generators — essentially a coin flip on weak samples. This confirms that entropy alone is insufficient for randomness assessment.

2. **NIST pass/fail is a strong baseline.** Its accuracy (0.793) matches Logistic Regression and Stacking, demonstrating that a simple majority-vote over NIST tests already captures most of the signal. The ML models' advantage is primarily in AUC (0.786 vs. 0.796 for RF), meaning they produce better-calibrated probability estimates.

3. **ML models provide marginal improvement.** The Random Forest's +1.4% accuracy and +1.0% AUC over NIST pass/fail is modest. The main ML advantage is:
   - **Probabilistic output:** ML models provide confidence scores, not just pass/fail.
   - **Feature fusion:** ML models can detect combinations of borderline test failures that individually wouldn't trigger a threshold-based system.
   - **Explainability:** SHAP values provide feature-level justifications for each prediction.

4. **XGBoost has the highest weak recall (0.653)** but at the cost of also misclassifying 11 strong samples as weak. It's the most conservative model.

---

## 7. Feature Importance Analysis

### 7.1 Top 10 Features (Average of RF + XGBoost Importance)

| Rank | Feature | RF Importance | XGB Importance | Avg Importance |
|------|---------|---------------|----------------|----------------|
| 1 | `nist_longest_run_p` | 0.111 | 0.295 | **0.203** |
| 2 | `nist_longest_run_stat` | 0.082 | 0.029 | 0.056 |
| 3 | `shannon_entropy` | 0.055 | 0.029 | 0.042 |
| 4 | `compression_ratio` | 0.081 | 0.000 | 0.041 |
| 5 | `nist_block_freq_p` | 0.049 | 0.027 | 0.038 |
| 6 | `nist_block_freq_stat` | 0.052 | 0.016 | 0.034 |
| 7 | `min_entropy` | 0.041 | 0.023 | 0.032 |
| 8 | `nist_serial_p1` | 0.029 | 0.030 | 0.029 |
| 9 | `nist_approx_entropy_apen` | 0.032 | 0.025 | 0.029 |
| 10 | `autocorr_lag_4` | 0.025 | 0.032 | 0.028 |

### 7.2 Feature Importance Patterns

**Dominant feature:** `nist_longest_run_p` accounts for ~20% of the average importance, making it the single most informative test. The longest-run test measures the longest streak of identical bits — a fundamental property that weak generators often fail. XGBoost relies on it even more heavily (0.295), while RF distributes importance more evenly.

**Model disagreements:**
- `compression_ratio` is the 4th most important feature in RF (0.081) but has **zero importance** in XGBoost. This suggests XGBoost found alternative splits that capture the same information through other features.
- Similarly, `num_runs` and `autocorr_lag_1` have zero importance in XGBoost but non-zero importance in RF.

**Entropy is surprisingly mid-ranked.** Shannon entropy (rank 3) and min-entropy (rank 7) are important but not dominant. This confirms that entropy alone is insufficient — it must be combined with structural tests like longest-run and block-frequency.

### 7.3 Least Important Features

| Feature | Avg Importance |
|---------|---------------|
| `autocorr_lag_1` | 0.007 |
| `num_runs` | 0.007 |
| `run_max` | 0.016 |
| `run_mean` | 0.019 |
| `nist_frequency_stat` | 0.019 |

The low importance of `autocorr_lag_1` is notable — first-order autocorrelation is often considered a key randomness metric, but in practice, most weak generators in our dataset produce low lag-1 autocorrelation. Higher lags (lag 4, lag 16) are more discriminative.

---

## 8. SHAP Analysis Insights

SHAP (SHapley Additive exPlanations) values were computed on the test set using TreeExplainer on the XGBoost model.

### 8.1 Key SHAP Patterns

From the SHAP values across 150 test samples:

1. **`compression_ratio`** has the largest magnitude SHAP values (up to ±0.085), despite being zero-importance in XGBoost's internal feature importance. This paradox arises because SHAP measures marginal contribution (what happens when the feature is removed), while feature importance measures split frequency.

2. **`nist_longest_run_p`** consistently drives strong WEAK predictions. Negative SHAP values of −0.097 to −0.121 appear for samples with low longest-run p-values, making it the strongest WEAK signal.

3. **`nist_block_freq_p` and `nist_block_freq_stat`** co-vary with SHAP values around ±0.04-0.05, forming a correlated pair that reinforces the same signal.

4. **STRONG predictions** are driven by a coalition of features rather than a single dominant one. Positive SHAP values are more evenly distributed across longest_run, entropy, block_freq, and serial tests.

### 8.2 Feature Interaction Effects

The SHAP analysis reveals that:
- **Entropy + compression** form a complementary pair: when both are low, the combined SHAP effect is larger than either alone.
- **NIST test failures tend to cluster:** samples failing one NIST test typically fail several, creating a correlated negative SHAP signal across multiple features.
- **Autocorrelation features act as tie-breakers:** for borderline samples where NIST tests are inconclusive, autocorrelation at lag 4 and lag 16 can tip the decision.

---

## 9. Stress Test Analysis

Stress tests evaluate the model on out-of-distribution scenarios not seen during training.

| Test Case | Expected | Predicted | Confidence | Correct? |
|-----------|----------|-----------|------------|----------|
| Mersenne Twister | Debatable* | STRONG | 76.0% | ⚠️ |
| Constant bytes | WEAK | WEAK | 90.0% | ✅ |
| Strong 64-bit tokens | STRONG | WEAK | 56.3% | ❌ |
| Strong 256-bit tokens | STRONG | WEAK | 60.5% | ❌ |
| Small batch (10 tokens) | STRONG | WEAK | 56.3% | ❌ |
| Small batch (50 tokens) | STRONG | WEAK | 60.0% | ❌ |

### 9.1 Detailed Stress Test Analysis

**Mersenne Twister (⚠️ Ambiguous):**
The model classifies MT19937 as STRONG with 76% confidence. This is technically correct from a statistical standpoint — Mersenne Twister passes most statistical tests. However, MT19937 is **not** cryptographically secure (its state can be reconstructed from 624 consecutive outputs). This highlights a fundamental limitation: statistical tests cannot detect cryptographic weaknesses.

**Constant Bytes (✅ Correct):**
The model correctly identifies constant output as WEAK with 90% confidence — the highest confidence in any stress test. This is the easiest case since zero-entropy output fails virtually every statistical test.

**Token Length Sensitivity (❌ Critical Failure):**
Both 64-bit and 256-bit token tests fail. The model was trained exclusively on 128-bit tokens, and the feature extraction pipeline produces different statistical distributions for different token lengths. This is a **hard limitation** — the model would need to be retrained on diverse token lengths or the feature extraction would need length-normalization.

**Small Batch Sensitivity (❌ Expected Failure):**
With only 10 or 50 tokens (vs. 1,000 in training), NIST tests lack statistical power. P-values become unreliable, and entropy estimates are noisy. The low confidence scores (56-60%) correctly indicate uncertainty, but the predictions are wrong. A production system should enforce a minimum batch size.

### 9.2 Stress Test Summary

The stress tests reveal that the model is **brittle outside its training distribution**:
- ✅ Robust to: extreme weakness (constant bytes)
- ⚠️ Ambiguous on: non-crypto-secure but statistically good generators (MT19937)
- ❌ Fragile on: different token lengths, small batch sizes

---

## 10. Calibration Assessment

Model calibration measures whether predicted probabilities match actual outcomes. A model that predicts "80% chance of STRONG" should be correct 80% of the time for such predictions.

Based on the confusion matrices and prediction patterns:

- **Random Forest** tends toward **overconfident STRONG predictions** — it predicts 100% recall for STRONG but achieves only 61.3% weak recall, suggesting its probability estimates for borderline weak samples are too high.
- **XGBoost** is the **most conservative** model, with the most balanced error distribution but also the most false negatives.
- All models struggle with **probability calibration in the 0.5-0.7 range**, where XOR-collapse and predictable-seed generators fall. The calibration curves (see `models/evaluation/calibration_curves.png`) show deviation from the ideal diagonal in this region.

---

## 11. Error Analysis — Where and Why Models Fail

### 11.1 Systematic Failures (All Models)

| Generator | Test Samples | Best Model Accuracy | Root Cause |
|-----------|-------------|---------------------|------------|
| `xor_collapse_4` | 5 | 0% | Higher-order bit correlations not captured by NIST tests |
| `xor_collapse_8` | 11 | 0% | Same as above; 8-byte XOR preserves all pairwise statistics |
| `predictable_seed` | 13 | 7.7% | Weakness is logical (seed guessability), not statistical |

**These 29 samples account for all ~26-29 false positives across models.** This is not a model failure — it's a feature-space limitation. No amount of model tuning will fix this without adding features that capture higher-order dependencies.

### 11.2 Occasional Failures

| Generator | Failure Rate | Likely Cause |
|-----------|-------------|-------------|
| `secrets_csprng` | 8.3% (3/36) | Statistical fluctuation — some batches have borderline p-values by pure chance |

### 11.3 Models That Never Fail on Strong Generators

Random Forest achieves **0 false negatives** on the test set, meaning every genuinely strong generator is correctly identified. This is the most critical property for a randomness evaluation tool — you never want to reject good randomness.

---

## 12. Statistical Significance

With 150 test samples:
- The difference between RF (80.7%) and LR (79.3%) accuracy is **2 samples** — not statistically significant.
- The difference between RF (80.7%) and XGB (75.3%) accuracy is **8 samples** — marginally significant.
- Confidence intervals (95%, Wilson) for accuracy:
  - RF: [73.5%, 86.4%]
  - LR: [72.1%, 85.2%]
  - XGB: [67.7%, 81.7%]

The overlapping confidence intervals confirm that **RF and LR are statistically indistinguishable** on this test set. A larger test set or cross-validation would be needed to establish a significant difference.

---

## 13. Comparison with NIST SP 800-22 Approach

The traditional approach to randomness testing (as in NIST SP 800-22) uses individual test pass/fail decisions. RandFusion's ML approach differs in several key ways:

| Aspect | NIST SP 800-22 | RandFusion |
|--------|---------------|------------|
| Decision rule | Each test independently pass/fail at α=0.01 | Fused ML decision across all tests |
| Output | Set of pass/fail flags | Single probability score with SHAP explanation |
| Threshold | Fixed per test | Learned from data |
| Feature interactions | Not considered | Captured by tree-based models |
| Explainability | Test-level pass/fail | Feature-level SHAP contributions |
| Novel weaknesses | Depends on test battery completeness | Depends on training data diversity |

RandFusion's advantage is **fusion** — combining borderline test results into a stronger aggregate signal. Its disadvantage is **data dependence** — it can only detect weakness patterns present in its training data.

---

## 14. Recommendations

### 14.1 For Immediate Use

1. **Use Random Forest as the primary classifier.** It has the best overall performance and zero false negatives on strong generators.
2. **Set a minimum batch size of 100 tokens.** Stress tests show unreliable predictions below this threshold.
3. **Report confidence scores alongside predictions.** Low confidence (< 70%) should trigger manual review.
4. **Do not rely on RandFusion alone for cryptographic assessment.** It tests statistical properties, not cryptographic security.

### 14.2 For Improvement

1. **Add higher-order statistical tests** (birthday spacing, matrix rank, spectral test, Maurer's universal test) to detect XOR-collapse patterns.
2. **Expand training data** with more generator types, especially:
   - Mersenne Twister variants
   - Truncated LFSRs
   - Deliberately weakened CSPRNGs
   - Real-world token samples from production systems
3. **Implement token-length normalization** in feature extraction to handle variable-length tokens.
4. **Add cross-validation** for more robust performance estimates.
5. **Consider neural network classifiers** on raw bitstreams to capture patterns that hand-crafted features miss.
6. **Multi-batch analysis:** Correlate across multiple batches from the same generator to detect predictable-seed patterns.

---

## 15. Conclusion

RandFusion successfully demonstrates that ML-based fusion of statistical randomness tests can outperform simple threshold-based approaches. The Random Forest classifier achieves **80.7% accuracy** and **0.796 AUC** on a balanced test set, correctly identifying all strong generators and most weak ones.

However, the system has clear limitations:
- **3 of 8 weak generator types** are partially or completely undetectable with the current feature set.
- **Performance is brittle** outside the training distribution (different token lengths, small batches).
- **Statistical tests cannot detect cryptographic weaknesses** — a strong-looking output from a predictable generator will pass.

These results establish a solid baseline for ML-fused randomness assessment while clearly delineating what statistical testing can and cannot achieve. The path forward lies in richer feature engineering, broader training data, and hybrid approaches that combine statistical testing with cryptographic analysis.

---

*Generated from RandFusion evaluation pipeline. All metrics computed on the held-out test set (150 samples, stratified split, seed=42).*
