"""
Tests for Phase 2: Statistical randomness tests and feature extraction.

Verifies:
  - Each NIST test returns valid p-values and statistics
  - Each non-NIST test returns expected keys
  - Strong randomness passes tests (high p-values)
  - Weak randomness fails at least some tests (low p-values)
  - FeatureExtractor produces consistent, complete feature vectors
"""

import numpy as np
import pytest
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
from src.features.extractor import FeatureExtractor


# Generate test bitstreams
np.random.seed(42)
RANDOM_BITS = np.random.randint(0, 2, size=128000).astype(np.uint8)  # 1000 tokens × 128 bits
BIASED_BITS = (np.random.random(128000) < 0.75).astype(np.uint8)     # 75% ones


class TestNISTTests:
    """Verify NIST test implementations return valid results."""

    def test_frequency_random(self):
        result = frequency_test(RANDOM_BITS)
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1
        # Random data should generally pass (p > 0.01)
        assert result["p_value"] > 0.01

    def test_frequency_biased(self):
        result = frequency_test(BIASED_BITS)
        # Biased data should fail
        assert result["p_value"] < 0.01

    def test_block_frequency(self):
        result = block_frequency_test(RANDOM_BITS, block_size=128)
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_runs(self):
        result = runs_test(RANDOM_BITS)
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_longest_run(self):
        result = longest_run_of_ones_test(RANDOM_BITS)
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_serial(self):
        result = serial_test(RANDOM_BITS)
        assert "p_value_1" in result
        assert "p_value_2" in result
        assert 0 <= result["p_value_1"] <= 1

    def test_approximate_entropy(self):
        result = approximate_entropy_test(RANDOM_BITS)
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_cumulative_sums(self):
        result = cumulative_sums_test(RANDOM_BITS)
        assert "p_value_forward" in result
        assert "p_value_backward" in result
        assert 0 <= result["p_value_forward"] <= 1


class TestStatisticalTests:
    """Verify non-NIST statistical tests."""

    def test_shannon_entropy_random(self):
        result = shannon_entropy(RANDOM_BITS)
        # Should be close to 8.0 for random bytes
        assert result["shannon_entropy"] > 7.5

    def test_shannon_entropy_biased(self):
        result = shannon_entropy(BIASED_BITS)
        # Biased → lower entropy
        assert result["shannon_entropy"] < 7.5

    def test_min_entropy_random(self):
        result = min_entropy(RANDOM_BITS)
        assert "min_entropy" in result
        assert result["min_entropy"] > 5.0  # should be near 8.0

    def test_run_length_stats(self):
        result = run_length_statistics(RANDOM_BITS)
        assert "run_mean" in result
        assert "run_std" in result
        assert "run_max" in result
        assert "num_runs" in result
        # Random bits: average run length ≈ 2.0
        assert 1.5 < result["run_mean"] < 2.5

    def test_autocorrelation_random(self):
        result = autocorrelation(RANDOM_BITS, [1, 2, 4])
        for lag in [1, 2, 4]:
            key = f"autocorr_lag_{lag}"
            assert key in result
            # Should be near zero for random data
            assert abs(result[key]) < 0.05

    def test_compression_ratio_random(self):
        result = compression_ratio(RANDOM_BITS)
        # Random data is incompressible: ratio should be near 1.0
        assert result["compression_ratio"] > 0.95


class TestFeatureExtractor:
    """Verify the FeatureExtractor wrapper."""

    def test_extract_returns_dict(self):
        fe = FeatureExtractor()
        batch = RANDOM_BITS.reshape(1000, 128)
        features = fe.extract(batch)
        assert isinstance(features, dict)
        assert len(features) > 20  # We expect ~28 features

    def test_all_values_are_float(self):
        fe = FeatureExtractor()
        batch = RANDOM_BITS.reshape(1000, 128)
        features = fe.extract(batch)
        for name, val in features.items():
            assert isinstance(val, (int, float)), f"{name} is {type(val)}"

    def test_no_nans_on_random_data(self):
        fe = FeatureExtractor()
        batch = RANDOM_BITS.reshape(1000, 128)
        features = fe.extract(batch)
        for name, val in features.items():
            assert not np.isnan(val), f"NaN in {name}"

    def test_feature_names_consistent(self):
        fe = FeatureExtractor()
        names = fe.feature_names()
        batch = RANDOM_BITS.reshape(1000, 128)
        features = fe.extract(batch)
        assert list(features.keys()) == names

    def test_num_features(self):
        fe = FeatureExtractor()
        n = fe.num_features()
        assert n > 20
        assert n == len(fe.feature_names())

    def test_weak_vs_strong_differ(self):
        """Weak and strong batches should produce different features."""
        fe = FeatureExtractor()
        strong = RANDOM_BITS.reshape(1000, 128)
        weak = BIASED_BITS.reshape(1000, 128)
        f_strong = fe.extract(strong)
        f_weak = fe.extract(weak)
        # At minimum, Shannon entropy should differ significantly
        assert abs(f_strong["shannon_entropy"] - f_weak["shannon_entropy"]) > 0.5
