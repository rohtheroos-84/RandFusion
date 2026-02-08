"""
Baseline 1 – Entropy Threshold Classifier.

Classifies a sample as weak (0) if its Shannon entropy is below a
threshold, and strong (1) otherwise.

The threshold is tuned on the training set by picking the value
that maximises the F1 score.
"""

import numpy as np
from sklearn.metrics import f1_score

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Feature column used by this baseline
ENTROPY_COLUMN = "shannon_entropy"


class EntropyThresholdClassifier:
    """Simple threshold classifier on Shannon entropy."""

    def __init__(self):
        self.threshold: float | None = None

    def fit(self, X_entropy: np.ndarray, y: np.ndarray) -> "EntropyThresholdClassifier":
        """Find the threshold that maximises F1 on the given data.

        Args:
            X_entropy: 1-D array of Shannon entropy values (one per sample).
            y: Ground-truth labels (0 = weak, 1 = strong).

        Returns:
            self (fitted).
        """
        # Search over candidate thresholds (unique values in the data)
        candidates = np.unique(X_entropy)
        # Add midpoints between adjacent candidates for finer search
        midpoints = (candidates[:-1] + candidates[1:]) / 2
        all_thresholds = np.sort(np.concatenate([candidates, midpoints]))

        best_f1 = -1.0
        best_thresh = float(candidates[0])

        for t in all_thresholds:
            y_pred = (X_entropy >= t).astype(int)
            score = f1_score(y, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = float(t)

        self.threshold = best_thresh
        logger.info(f"EntropyThreshold fitted: threshold={self.threshold:.6f}, "
                    f"train F1={best_f1:.4f}")
        return self

    def predict(self, X_entropy: np.ndarray) -> np.ndarray:
        """Predict labels using the fitted threshold.

        Args:
            X_entropy: 1-D array of Shannon entropy values.

        Returns:
            Array of predicted labels (0 or 1).
        """
        if self.threshold is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        return (X_entropy >= self.threshold).astype(int)

    def predict_proba(self, X_entropy: np.ndarray) -> np.ndarray:
        """Return pseudo-probabilities based on distance from threshold.

        Maps entropy values to [0, 1] using a sigmoid-like rescaling
        centred on the threshold so that ROC-AUC can be computed.

        Args:
            X_entropy: 1-D array of Shannon entropy values.

        Returns:
            2-D array of shape (n_samples, 2) with columns [P(weak), P(strong)].
        """
        if self.threshold is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        # Centre & scale: distance from threshold, normalised by data range
        diff = X_entropy - self.threshold
        # Use a simple sigmoid-like mapping
        # Scale factor chosen so most values spread across [0.1, 0.9]
        scale = max(np.std(diff), 1e-9)
        prob_strong = 1.0 / (1.0 + np.exp(-diff / scale))

        proba = np.column_stack([1 - prob_strong, prob_strong])
        return proba
