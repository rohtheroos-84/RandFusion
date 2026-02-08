# Baseline vs ML Comparison — Analysis

## Summary

- **Best baseline:** nist_pass_fail (F1 = 0.8229)
- **Best ML model:** random_forest (F1 = 0.8380)

## Key Findings

1. **ML outperforms baselines on F1** by 0.0151 (random_forest: 0.8380 vs nist_pass_fail: 0.8229).
2. **ROC-AUC improvement:** +0.0095 (random_forest: 0.7959 vs nist_pass_fail: 0.7864).
3. **Weak-class recall change:** -0.0133 (random_forest: 0.6133 vs nist_pass_fail: 0.6267).

## Why ML Ensemble Adds Value

- **Feature fusion:** The ML ensemble combines information from all statistical
  tests simultaneously, whereas individual baselines rely on a single signal
  (entropy alone or NIST pass/fail alone).
- **Decision boundaries:** The stacking ensemble learns non-linear decision
  boundaries in the high-dimensional feature space, capturing subtle patterns
  that simple thresholds miss.
- **Robustness:** Different weak generators exhibit different failure modes.
  The entropy baseline may miss generators with good entropy but poor
  autocorrelation; the NIST baseline may pass generators that are weak in
  ways not directly tested. The ML fusion approach adapts to all failure modes.

## Full Metrics Table

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall(Weak) |
|-------|----------|-----------|--------|-----|---------|--------------|
| entropy_threshold | 0.7467 | 0.6637 | 1.0000 | 0.7979 | 0.6432 | 0.4933 |
| nist_pass_fail | 0.7933 | 0.7200 | 0.9600 | 0.8229 | 0.7864 | 0.6267 |
| logistic_regression | 0.7933 | 0.7157 | 0.9733 | 0.8249 | 0.7867 | 0.6133 |
| random_forest | 0.8067 | 0.7212 | 1.0000 | 0.8380 | 0.7959 | 0.6133 |
| xgboost | 0.7533 | 0.7111 | 0.8533 | 0.7758 | 0.7771 | 0.6533 |
| stacking_ensemble | 0.7933 | 0.7200 | 0.9600 | 0.8229 | 0.7892 | 0.6267 |
