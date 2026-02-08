"""
Tests for Phase 3: Feature dataset construction.

Verifies:
  - Feature CSV exists and has correct shape
  - No NaN or Inf values
  - Labels are balanced
  - Feature names are consistent with FeatureExtractor
  - Strong and weak samples are separable on key features
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.utils.config import PROJECT_ROOT, get_config
from src.features.extractor import FeatureExtractor


@pytest.fixture
def feature_df():
    config = get_config()
    csv_path = PROJECT_ROOT / config["dataset"]["processed_dir"] / "features.csv"
    return pd.read_csv(csv_path)


class TestFeatureDataset:

    def test_file_exists(self):
        config = get_config()
        csv_path = PROJECT_ROOT / config["dataset"]["processed_dir"] / "features.csv"
        assert csv_path.exists()

    def test_shape(self, feature_df):
        # 1000 samples, 32 features + label + generator = 34 columns
        assert feature_df.shape[0] == 1000
        assert feature_df.shape[1] == 34

    def test_no_nans(self, feature_df):
        fe = FeatureExtractor()
        feature_cols = fe.feature_names()
        assert feature_df[feature_cols].isna().sum().sum() == 0

    def test_no_infs(self, feature_df):
        fe = FeatureExtractor()
        feature_cols = fe.feature_names()
        assert not np.any(np.isinf(feature_df[feature_cols].values))

    def test_balanced_labels(self, feature_df):
        counts = feature_df["label"].value_counts()
        assert counts[0] == 500
        assert counts[1] == 500

    def test_feature_columns_match_extractor(self, feature_df):
        fe = FeatureExtractor()
        expected = fe.feature_names()
        actual = [c for c in feature_df.columns if c not in ("label", "generator")]
        assert actual == expected

    def test_strong_weak_entropy_separable(self, feature_df):
        """Shannon entropy should clearly differ between classes."""
        strong = feature_df.loc[feature_df["label"] == 1, "shannon_entropy"]
        weak = feature_df.loc[feature_df["label"] == 0, "shannon_entropy"]
        assert strong.mean() > weak.mean()

    def test_generators_present(self, feature_df):
        """All 10 generator types should be present."""
        assert feature_df["generator"].nunique() == 10
