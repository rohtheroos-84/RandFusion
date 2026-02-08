"""
Feature extractor for RandFusion.

Combines all statistical tests into a single class that takes a token
batch (2D array: batch_size × token_bits) and returns one feature vector
(a dict of named numerical features).

The batch is first concatenated into one long bitstream, then all tests
are run on that bitstream.

Usage:
    from src.features.extractor import FeatureExtractor
    fe = FeatureExtractor()
    features = fe.extract(batch)  # dict of floats
    print(fe.feature_names())     # list of feature names
"""

import numpy as np
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.features.nist_tests import (
    frequency_test,
    block_frequency_test,
    runs_test,
    longest_run_of_ones_test,
    serial_test,
    approximate_entropy_test,
    cumulative_sums_test,
)
from src.features.statistical_tests import (
    shannon_entropy,
    min_entropy,
    run_length_statistics,
    autocorrelation,
    compression_ratio,
)

logger = get_logger(__name__)


class FeatureExtractor:
    """Extracts a fixed-length feature vector from a token batch.

    All features are numerical (float). The feature names are consistent
    and documented, so the ML model always sees the same columns.
    """

    def __init__(self, config: dict = None):
        """Initialize with configuration parameters.

        Args:
            config: Configuration dict. If None, loads from default config.yaml.
        """
        if config is None:
            config = get_config()

        feat_cfg = config.get("features", {})
        self.nist_block_size = feat_cfg.get("nist_block_size", 128)
        self.autocorrelation_lags = feat_cfg.get("autocorrelation_lags", [1, 2, 4, 8, 16])
        self.compression_level = feat_cfg.get("compression_level", 6)

        # Cache the feature names list (computed once)
        self._feature_names = None

    def extract(self, batch: np.ndarray) -> dict:
        """Extract all features from a token batch.

        Args:
            batch: 2D array of shape (num_tokens, token_length_bits), values 0/1.

        Returns:
            Dictionary mapping feature name → float value.
        """
        # Concatenate all tokens into one long bitstream
        bitstream = batch.flatten()

        features = {}

        # --- NIST Tests ---
        try:
            result = frequency_test(bitstream)
            features["nist_frequency_p"] = result["p_value"]
            features["nist_frequency_stat"] = result["statistic"]
        except Exception as e:
            logger.warning(f"Frequency test failed: {e}")
            features["nist_frequency_p"] = 0.0
            features["nist_frequency_stat"] = float("nan")

        try:
            result = block_frequency_test(bitstream, self.nist_block_size)
            features["nist_block_freq_p"] = result["p_value"]
            features["nist_block_freq_stat"] = result["statistic"]
        except Exception as e:
            logger.warning(f"Block frequency test failed: {e}")
            features["nist_block_freq_p"] = 0.0
            features["nist_block_freq_stat"] = float("nan")

        try:
            result = runs_test(bitstream)
            features["nist_runs_p"] = result["p_value"]
            features["nist_runs_stat"] = result["statistic"]
        except Exception as e:
            logger.warning(f"Runs test failed: {e}")
            features["nist_runs_p"] = 0.0
            features["nist_runs_stat"] = float("nan")

        try:
            result = longest_run_of_ones_test(bitstream)
            features["nist_longest_run_p"] = result["p_value"]
            features["nist_longest_run_stat"] = result["statistic"]
        except Exception as e:
            logger.warning(f"Longest run test failed: {e}")
            features["nist_longest_run_p"] = 0.0
            features["nist_longest_run_stat"] = float("nan")

        try:
            result = serial_test(bitstream, block_length=3)
            features["nist_serial_p1"] = result["p_value_1"]
            features["nist_serial_p2"] = result["p_value_2"]
            features["nist_serial_delta1"] = result["statistic_delta1"]
            features["nist_serial_delta2"] = result["statistic_delta2"]
        except Exception as e:
            logger.warning(f"Serial test failed: {e}")
            features["nist_serial_p1"] = 0.0
            features["nist_serial_p2"] = 0.0
            features["nist_serial_delta1"] = float("nan")
            features["nist_serial_delta2"] = float("nan")

        try:
            result = approximate_entropy_test(bitstream, block_length=3)
            features["nist_approx_entropy_p"] = result["p_value"]
            features["nist_approx_entropy_stat"] = result["statistic"]
            features["nist_approx_entropy_apen"] = result["apen"]
        except Exception as e:
            logger.warning(f"Approximate entropy test failed: {e}")
            features["nist_approx_entropy_p"] = 0.0
            features["nist_approx_entropy_stat"] = float("nan")
            features["nist_approx_entropy_apen"] = float("nan")

        try:
            result = cumulative_sums_test(bitstream)
            features["nist_cusum_fwd_p"] = result["p_value_forward"]
            features["nist_cusum_fwd_stat"] = result["statistic_forward"]
            features["nist_cusum_bwd_p"] = result["p_value_backward"]
            features["nist_cusum_bwd_stat"] = result["statistic_backward"]
        except Exception as e:
            logger.warning(f"Cumulative sums test failed: {e}")
            features["nist_cusum_fwd_p"] = 0.0
            features["nist_cusum_fwd_stat"] = float("nan")
            features["nist_cusum_bwd_p"] = 0.0
            features["nist_cusum_bwd_stat"] = float("nan")

        # --- Entropy measures ---
        try:
            result = shannon_entropy(bitstream)
            features["shannon_entropy"] = result["shannon_entropy"]
        except Exception as e:
            logger.warning(f"Shannon entropy failed: {e}")
            features["shannon_entropy"] = 0.0

        try:
            result = min_entropy(bitstream)
            features["min_entropy"] = result["min_entropy"]
            features["max_probability"] = result["max_probability"]
        except Exception as e:
            logger.warning(f"Min-entropy failed: {e}")
            features["min_entropy"] = 0.0
            features["max_probability"] = 1.0

        # --- Run-length statistics ---
        try:
            result = run_length_statistics(bitstream)
            features["run_mean"] = result["run_mean"]
            features["run_std"] = result["run_std"]
            features["run_max"] = float(result["run_max"])
            features["num_runs"] = float(result["num_runs"])
        except Exception as e:
            logger.warning(f"Run-length stats failed: {e}")
            features["run_mean"] = 0.0
            features["run_std"] = 0.0
            features["run_max"] = 0.0
            features["num_runs"] = 0.0

        # --- Autocorrelation ---
        try:
            result = autocorrelation(bitstream, self.autocorrelation_lags)
            features.update(result)
        except Exception as e:
            logger.warning(f"Autocorrelation failed: {e}")
            for lag in self.autocorrelation_lags:
                features[f"autocorr_lag_{lag}"] = 0.0

        # --- Compression ---
        try:
            result = compression_ratio(bitstream, self.compression_level)
            features["compression_ratio"] = result["compression_ratio"]
        except Exception as e:
            logger.warning(f"Compression ratio failed: {e}")
            features["compression_ratio"] = 1.0

        return features

    def feature_names(self) -> list[str]:
        """Return the ordered list of feature names.

        This is computed by running extract() on a dummy batch once
        and caching the result. Guarantees consistent column ordering.
        """
        if self._feature_names is None:
            dummy = np.random.randint(0, 2, size=(10, 128)).astype(np.uint8)
            dummy_features = self.extract(dummy)
            self._feature_names = list(dummy_features.keys())
        return self._feature_names

    def num_features(self) -> int:
        """Return the total number of features extracted."""
        return len(self.feature_names())
