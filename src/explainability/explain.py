"""
explain_decision() — Human-readable explanation for a single sample.

Given a token batch (or pre-extracted feature vector), returns a
structured explanation including:
  - Predicted class + confidence
  - Top contributing features (positive and negative)
  - Natural-language summary sentence

Also provides a batch version for multiple samples.
"""

import joblib
import numpy as np
import shap

from pathlib import Path
from dataclasses import dataclass, field

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logger import get_logger
from src.features.extractor import FeatureExtractor

logger = get_logger(__name__)


@dataclass
class Explanation:
    """Structured explanation for a single randomness verdict."""
    predicted_class: int                 # 0 = weak, 1 = strong
    predicted_label: str                 # "STRONG" or "WEAK"
    confidence: float                    # P(predicted_class)
    top_positive_features: list[dict]    # Features pushing toward predicted class
    top_negative_features: list[dict]    # Features pushing against predicted class
    summary: str                         # One-sentence natural-language summary
    shap_values: dict                    # All SHAP values as {feature: value}


class DecisionExplainer:
    """Loads models and provides per-sample explanations.

    Uses the Random Forest model with SHAP TreeExplainer for fast,
    exact explanations.
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = get_config()

        models_dir = PROJECT_ROOT / config["model"]["output_dir"]
        self.rf_model = joblib.load(models_dir / "random_forest.joblib")
        self.feature_extractor = FeatureExtractor(config)
        self.feature_names = self.feature_extractor.feature_names()

        # Build SHAP explainer (lazy, created on first explain call)
        self._explainer = None
        self._config = config

    def _get_explainer(self) -> shap.TreeExplainer:
        """Lazily create the SHAP TreeExplainer."""
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.rf_model)
        return self._explainer

    def explain_features(
        self,
        feature_vector: np.ndarray,
        top_k: int = 5,
    ) -> Explanation:
        """Explain a decision given a pre-extracted feature vector.

        Args:
            feature_vector: 1-D array of shape (n_features,) matching
                            self.feature_names ordering.
            top_k: Number of top contributing features to return.

        Returns:
            Explanation dataclass.
        """
        X = feature_vector.reshape(1, -1)

        # Prediction
        proba = self.rf_model.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])
        pred_label = "STRONG" if pred_class == 1 else "WEAK"

        # SHAP values
        explainer = self._get_explainer()
        shap_result = explainer(X)

        # Handle multi-output SHAP (binary classifier)
        sv = shap_result.values[0]
        if sv.ndim > 1:
            sv = sv[:, 1]  # SHAP values for class 1 (strong)

        shap_dict = {name: float(val) for name, val in zip(self.feature_names, sv)}

        # Sort features by absolute SHAP value
        sorted_features = sorted(
            shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Split into positive (pushing toward strong) and negative (toward weak)
        positive = [
            {"feature": name, "shap_value": val, "direction": "-> STRONG"}
            for name, val in sorted_features if val > 0
        ][:top_k]

        negative = [
            {"feature": name, "shap_value": val, "direction": "-> WEAK"}
            for name, val in sorted_features if val <= 0
        ][:top_k]

        # Natural-language summary
        summary = self._build_summary(
            pred_label, confidence, positive, negative, feature_vector
        )

        return Explanation(
            predicted_class=pred_class,
            predicted_label=pred_label,
            confidence=confidence,
            top_positive_features=positive,
            top_negative_features=negative,
            summary=summary,
            shap_values=shap_dict,
        )

    def explain_batch(
        self,
        batch: np.ndarray,
        top_k: int = 5,
    ) -> Explanation:
        """Explain a decision for a raw token batch.

        Args:
            batch: 2-D array of shape (num_tokens, token_bits), values 0/1.
            top_k: Number of top contributing features to return.

        Returns:
            Explanation dataclass.
        """
        features = self.feature_extractor.extract(batch)
        feature_vector = np.array([features[name] for name in self.feature_names])
        return self.explain_features(feature_vector, top_k=top_k)

    def _build_summary(
        self,
        pred_label: str,
        confidence: float,
        positive: list[dict],
        negative: list[dict],
        feature_vector: np.ndarray,
    ) -> str:
        """Build a one-sentence natural-language summary."""
        conf_pct = confidence * 100

        if pred_label == "STRONG":
            # Top reasons it's strong
            if positive:
                top_feature = positive[0]["feature"]
                reason = _feature_explanation(top_feature)
                return (
                    f"The token batch is classified as STRONG (confidence: {conf_pct:.1f}%). "
                    f"The most influential factor is {reason}. "
                    f"Overall, {len(positive)} features support this verdict."
                )
            return f"The token batch is classified as STRONG (confidence: {conf_pct:.1f}%)."
        else:
            # Top reasons it's weak
            if negative:
                top_feature = negative[0]["feature"]
                reason = _feature_explanation(top_feature)
                return (
                    f"The token batch is classified as WEAK (confidence: {conf_pct:.1f}%). "
                    f"The most influential factor is {reason}. "
                    f"Overall, {len(negative)} features flag potential weaknesses."
                )
            return f"The token batch is classified as WEAK (confidence: {conf_pct:.1f}%)."


def _feature_explanation(feature_name: str) -> str:
    """Return a human-readable explanation of what a feature measures."""
    explanations = {
        "nist_frequency_p": "the NIST monobit frequency test p-value (overall bit balance)",
        "nist_frequency_stat": "the NIST frequency test statistic",
        "nist_block_freq_p": "the NIST block frequency test p-value (local bit balance)",
        "nist_block_freq_stat": "the NIST block frequency chi-squared statistic",
        "nist_runs_p": "the NIST runs test p-value (transitions between 0s and 1s)",
        "nist_runs_stat": "the number of runs in the bitstream",
        "nist_longest_run_p": "the NIST longest-run-of-ones test p-value",
        "nist_longest_run_stat": "the longest-run chi-squared statistic",
        "nist_serial_p1": "the NIST serial test p-value (pattern uniformity)",
        "nist_serial_p2": "the NIST serial test second p-value",
        "nist_serial_delta1": "the serial test delta-1 statistic",
        "nist_serial_delta2": "the serial test delta-2 statistic",
        "nist_approx_entropy_p": "the NIST approximate entropy test p-value (pattern predictability)",
        "nist_approx_entropy_stat": "the approximate entropy chi-squared statistic",
        "nist_approx_entropy_apen": "the approximate entropy (ApEn) value",
        "nist_cusum_fwd_p": "the NIST cumulative sums (forward) p-value",
        "nist_cusum_fwd_stat": "the cumulative sums forward statistic",
        "nist_cusum_bwd_p": "the NIST cumulative sums (backward) p-value",
        "nist_cusum_bwd_stat": "the cumulative sums backward statistic",
        "shannon_entropy": "the Shannon entropy of byte distribution",
        "min_entropy": "the min-entropy (worst-case unpredictability)",
        "max_probability": "the maximum byte probability (uniformity measure)",
        "run_mean": "the mean run length (average consecutive same-bit streak)",
        "run_std": "the run length standard deviation",
        "run_max": "the maximum run length (longest consecutive same-bit streak)",
        "num_runs": "the total number of runs",
        "autocorr_lag_1": "the autocorrelation at lag 1 (sequential dependency)",
        "autocorr_lag_2": "the autocorrelation at lag 2",
        "autocorr_lag_4": "the autocorrelation at lag 4",
        "autocorr_lag_8": "the autocorrelation at lag 8",
        "autocorr_lag_16": "the autocorrelation at lag 16",
        "compression_ratio": "the compression ratio (compressibility of the bitstream)",
    }
    return explanations.get(feature_name, f"the '{feature_name}' metric")


def format_explanation(exp: Explanation) -> str:
    """Format an Explanation as a printable multiline string."""
    lines = [
        "=" * 60,
        f"  VERDICT: {exp.predicted_label}  "
        f"(confidence: {exp.confidence * 100:.1f}%)",
        "=" * 60,
        "",
        exp.summary,
        "",
        "Top features SUPPORTING the verdict:",
    ]

    for feat in exp.top_positive_features:
        lines.append(
            f"   {feat['direction']}  {feat['feature']:30s}  "
            f"SHAP = {feat['shap_value']:+.4f}"
        )

    lines.append("")
    lines.append("Top features OPPOSING the verdict:")

    for feat in exp.top_negative_features:
        lines.append(
            f"   {feat['direction']}  {feat['feature']:30s}  "
            f"SHAP = {feat['shap_value']:+.4f}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
