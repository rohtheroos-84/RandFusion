"""
Tests for Phase 5 — Baseline Comparison.

Covers:
  - EntropyThresholdClassifier: fit, predict, predict_proba, edge cases
  - NistPassFailClassifier: predict, predict_proba, edge cases
  - Integration: both baselines produce valid metrics on the real feature CSV
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.baselines.entropy_threshold import EntropyThresholdClassifier, ENTROPY_COLUMN
from src.baselines.nist_pass_fail import NistPassFailClassifier, NIST_P_COLUMNS
from src.utils.config import get_config, PROJECT_ROOT
from src.features.extractor import FeatureExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def entropy_data():
    """Synthetic entropy data: strong samples have high entropy, weak low."""
    rng = np.random.RandomState(42)
    n = 100
    # Strong: entropy ~ 7.99; Weak: entropy ~ 7.5
    strong = rng.normal(loc=7.99, scale=0.002, size=n // 2)
    weak = rng.normal(loc=7.50, scale=0.10, size=n // 2)
    X = np.concatenate([strong, weak])
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    return X, y


@pytest.fixture
def nist_data():
    """Synthetic NIST p-value data."""
    rng = np.random.RandomState(42)
    n = 100
    n_tests = len(NIST_P_COLUMNS)
    # Strong: all p-values high; Weak: at least one very low
    strong = rng.uniform(0.1, 1.0, size=(n // 2, n_tests))
    weak = rng.uniform(0.1, 1.0, size=(n // 2, n_tests))
    # Make sure weak has at least one failing p-value per sample
    weak[:, 0] = rng.uniform(0.0001, 0.005, size=n // 2)
    X = np.vstack([strong, weak])
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    return X, y


# ---------------------------------------------------------------------------
# EntropyThresholdClassifier tests
# ---------------------------------------------------------------------------

class TestEntropyThreshold:

    def test_fit_sets_threshold(self, entropy_data):
        X, y = entropy_data
        clf = EntropyThresholdClassifier()
        clf.fit(X, y)
        assert clf.threshold is not None
        assert isinstance(clf.threshold, float)

    def test_predict_returns_binary(self, entropy_data):
        X, y = entropy_data
        clf = EntropyThresholdClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})
        assert preds.shape == y.shape

    def test_predict_proba_shape_and_range(self, entropy_data):
        X, y = entropy_data
        clf = EntropyThresholdClassifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        # Probabilities should sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-7)

    def test_separable_data_high_accuracy(self, entropy_data):
        X, y = entropy_data
        clf = EntropyThresholdClassifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = np.mean(preds == y)
        # With well-separated data, should get >90% accuracy
        assert acc > 0.90

    def test_predict_before_fit_raises(self):
        clf = EntropyThresholdClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(np.array([7.5, 7.9]))

    def test_predict_proba_before_fit_raises(self):
        clf = EntropyThresholdClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(np.array([7.5, 7.9]))


# ---------------------------------------------------------------------------
# NistPassFailClassifier tests
# ---------------------------------------------------------------------------

class TestNistPassFail:

    def test_predict_returns_binary(self, nist_data):
        X, y = nist_data
        clf = NistPassFailClassifier(alpha=0.01)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})
        assert preds.shape == y.shape

    def test_all_pass_classified_strong(self):
        """If all p-values are above alpha, sample should be strong."""
        X = np.full((5, len(NIST_P_COLUMNS)), 0.5)  # All well above 0.01
        clf = NistPassFailClassifier(alpha=0.01)
        preds = clf.predict(X)
        np.testing.assert_array_equal(preds, 1)

    def test_any_fail_classified_weak(self):
        """If any single p-value is below alpha, sample should be weak."""
        X = np.full((5, len(NIST_P_COLUMNS)), 0.5)
        X[:, 3] = 0.001  # One failing test
        clf = NistPassFailClassifier(alpha=0.01)
        preds = clf.predict(X)
        np.testing.assert_array_equal(preds, 0)

    def test_predict_proba_shape_and_range(self, nist_data):
        X, y = nist_data
        clf = NistPassFailClassifier(alpha=0.01)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-7)

    def test_different_alpha_changes_predictions(self, nist_data):
        X, y = nist_data
        # Add borderline values
        X[0, 0] = 0.05  # Would pass at alpha=0.01, fail at alpha=0.1
        clf_strict = NistPassFailClassifier(alpha=0.01)
        clf_lenient = NistPassFailClassifier(alpha=0.1)
        pred_strict = clf_strict.predict(X)
        pred_lenient = clf_lenient.predict(X)
        # Lenient alpha should classify more as weak
        assert pred_lenient.sum() <= pred_strict.sum()


# ---------------------------------------------------------------------------
# Integration test on real data
# ---------------------------------------------------------------------------

class TestBaselineIntegration:

    @pytest.fixture
    def real_data(self):
        """Load the actual feature CSV and split like train.py."""
        config = get_config()
        csv_path = PROJECT_ROOT / config["dataset"]["processed_dir"] / "features.csv"
        if not csv_path.exists():
            pytest.skip("features.csv not found — run Phase 3 first")
        df = pd.read_csv(csv_path)
        fe = FeatureExtractor(config)
        return df, fe.feature_names()

    def test_entropy_baseline_on_real_data(self, real_data):
        df, feature_names = real_data
        X = df[feature_names].values
        y = df["label"].values

        ent_idx = feature_names.index(ENTROPY_COLUMN)
        clf = EntropyThresholdClassifier()
        clf.fit(X[:, ent_idx], y)
        preds = clf.predict(X[:, ent_idx])

        # Should produce reasonable accuracy (better than random)
        acc = np.mean(preds == y)
        assert acc > 0.50, f"Accuracy {acc} not better than random"

    def test_nist_baseline_on_real_data(self, real_data):
        df, feature_names = real_data
        X = df[feature_names].values
        y = df["label"].values

        nist_idxs = [feature_names.index(c) for c in NIST_P_COLUMNS]
        clf = NistPassFailClassifier(alpha=0.01)
        clf.fit(X[:, nist_idxs], y)
        preds = clf.predict(X[:, nist_idxs])

        acc = np.mean(preds == y)
        assert acc > 0.50, f"Accuracy {acc} not better than random"

    def test_comparison_results_file_exists(self):
        """After running compare(), the results JSON should exist."""
        config = get_config()
        results_path = (PROJECT_ROOT / config["model"]["output_dir"]
                        / "comparison" / "comparison_results.json")
        # This test is gated: skip if not yet run
        if not results_path.exists():
            pytest.skip("comparison_results.json not found — run compare first")
        with open(results_path) as f:
            data = json.load(f)
        assert "entropy_threshold" in data
        assert "nist_pass_fail" in data
        assert len(data) >= 4  # 2 baselines + at least 2 ML models


import json
