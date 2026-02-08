"""
Baseline 2 – NIST All-Tests-Pass Classifier.

Classifies a sample as weak (0) if *any* NIST SP 800-22 p-value falls
below a significance threshold (default 0.01), and strong (1) otherwise.

This mirrors the standard usage of NIST tests: a sample "fails" if any
single test rejects the null hypothesis of randomness.
"""

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# All NIST p-value columns in the feature CSV
NIST_P_COLUMNS = [
    "nist_frequency_p",
    "nist_block_freq_p",
    "nist_runs_p",
    "nist_longest_run_p",
    "nist_serial_p1",
    "nist_serial_p2",
    "nist_approx_entropy_p",
    "nist_cusum_fwd_p",
    "nist_cusum_bwd_p",
]


class NistPassFailClassifier:
    """Rule-based classifier: weak if any NIST p-value < alpha."""

    def __init__(self, alpha: float = 0.01):
        """
        Args:
            alpha: Significance level. Any p-value below this triggers
                   a 'weak' classification. Default 0.01.
        """
        self.alpha = alpha

    def fit(self, X_nist: np.ndarray, y: np.ndarray) -> "NistPassFailClassifier":
        """No-op: this baseline has no trainable parameters.

        Accepts the same interface as the entropy classifier for consistency.

        Args:
            X_nist: 2-D array of NIST p-values (n_samples × n_tests).
            y: Ground-truth labels (unused).

        Returns:
            self.
        """
        y_pred = self.predict(X_nist)
        from sklearn.metrics import f1_score
        score = f1_score(y, y_pred, zero_division=0)
        logger.info(f"NistPassFail (alpha={self.alpha}): train F1={score:.4f}")
        return self

    def predict(self, X_nist: np.ndarray) -> np.ndarray:
        """Predict labels based on NIST p-values.

        Args:
            X_nist: 2-D array of shape (n_samples, n_nist_tests).
                    Each column is a p-value from a different NIST test.

        Returns:
            Array of predicted labels: 0 (weak) if any p-value < alpha,
            1 (strong) otherwise.
        """
        # Weak if ANY test fails (p < alpha)
        any_fail = np.any(X_nist < self.alpha, axis=1)
        return (~any_fail).astype(int)

    def predict_proba(self, X_nist: np.ndarray) -> np.ndarray:
        """Return pseudo-probabilities for ROC-AUC computation.

        Uses the minimum p-value across all NIST tests as a continuous
        score. Higher min-p → more likely strong.

        Args:
            X_nist: 2-D array of shape (n_samples, n_nist_tests).

        Returns:
            2-D array of shape (n_samples, 2) with columns [P(weak), P(strong)].
        """
        # Min p-value across all tests: a natural "randomness score"
        min_p = np.min(X_nist, axis=1)
        # Clip to [0, 1] and use directly as P(strong)
        prob_strong = np.clip(min_p, 0.0, 1.0)

        proba = np.column_stack([1 - prob_strong, prob_strong])
        return proba
