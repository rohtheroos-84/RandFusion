"""
Tests for Phase 6 — Explainability.

Covers:
  - Feature importance computation and plots
  - SHAP value computation
  - DecisionExplainer: explain_features, explain formatting
  - Integration: full pipeline produces expected outputs
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.utils.config import get_config, PROJECT_ROOT
from src.features.extractor import FeatureExtractor


# ---------------------------------------------------------------------------
# Feature Importance tests
# ---------------------------------------------------------------------------

class TestFeatureImportance:

    def test_compute_importances_returns_dataframe(self):
        """compute_importances returns a DataFrame with expected columns."""
        from src.explainability.feature_importance import compute_importances, load_feature_names

        config = get_config()
        models_dir = PROJECT_ROOT / config["model"]["output_dir"]
        feature_names = load_feature_names(config)

        df = compute_importances(models_dir, feature_names)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "rf_importance" in df.columns
        assert "xgb_importance" in df.columns
        assert "avg_importance" in df.columns
        assert len(df) == len(feature_names)

    def test_importances_are_nonnegative(self):
        """All importance values should be >= 0."""
        from src.explainability.feature_importance import compute_importances, load_feature_names

        config = get_config()
        models_dir = PROJECT_ROOT / config["model"]["output_dir"]
        feature_names = load_feature_names(config)

        df = compute_importances(models_dir, feature_names)
        assert (df["rf_importance"] >= 0).all()
        assert (df["xgb_importance"] >= 0).all()

    def test_importances_sorted_descending(self):
        """DataFrame should be sorted by avg_importance descending."""
        from src.explainability.feature_importance import compute_importances, load_feature_names

        config = get_config()
        models_dir = PROJECT_ROOT / config["model"]["output_dir"]
        feature_names = load_feature_names(config)

        df = compute_importances(models_dir, feature_names)
        assert df["avg_importance"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# DecisionExplainer tests
# ---------------------------------------------------------------------------

class TestDecisionExplainer:

    @pytest.fixture
    def explainer(self):
        from src.explainability.explain import DecisionExplainer
        return DecisionExplainer()

    @pytest.fixture
    def test_features(self):
        """Load a real feature vector from the dataset."""
        config = get_config()
        csv_path = PROJECT_ROOT / config["dataset"]["processed_dir"] / "features.csv"
        df = pd.read_csv(csv_path)
        fe = FeatureExtractor(config)
        feature_names = fe.feature_names()
        return df[feature_names].values[0], df["label"].values[0]

    def test_explain_returns_explanation(self, explainer, test_features):
        """explain_features returns a valid Explanation dataclass."""
        from src.explainability.explain import Explanation

        fv, true_label = test_features
        exp = explainer.explain_features(fv, top_k=5)

        assert isinstance(exp, Explanation)
        assert exp.predicted_class in (0, 1)
        assert exp.predicted_label in ("STRONG", "WEAK")
        assert 0.0 <= exp.confidence <= 1.0

    def test_explanation_has_top_features(self, explainer, test_features):
        """Explanation should include top positive and negative features."""
        fv, _ = test_features
        exp = explainer.explain_features(fv, top_k=3)

        # At least one direction should have features
        assert len(exp.top_positive_features) > 0 or len(exp.top_negative_features) > 0
        # At most top_k per direction
        assert len(exp.top_positive_features) <= 3
        assert len(exp.top_negative_features) <= 3

    def test_explanation_has_summary(self, explainer, test_features):
        """Explanation should include a non-empty summary string."""
        fv, _ = test_features
        exp = explainer.explain_features(fv)

        assert isinstance(exp.summary, str)
        assert len(exp.summary) > 20
        assert exp.predicted_label in exp.summary

    def test_explanation_shap_values_dict(self, explainer, test_features):
        """SHAP values dict should have all feature names."""
        fv, _ = test_features
        exp = explainer.explain_features(fv)

        assert isinstance(exp.shap_values, dict)
        assert len(exp.shap_values) == len(explainer.feature_names)

    def test_format_explanation(self, explainer, test_features):
        """format_explanation produces a printable string."""
        from src.explainability.explain import format_explanation

        fv, _ = test_features
        exp = explainer.explain_features(fv)
        text = format_explanation(exp)

        assert isinstance(text, str)
        assert "VERDICT" in text
        assert exp.predicted_label in text


# ---------------------------------------------------------------------------
# SHAP Analysis tests
# ---------------------------------------------------------------------------

class TestShapAnalysis:

    def test_compute_shap_values_shape(self):
        """SHAP values should have correct shape."""
        import shap
        from src.explainability.shap_analysis import compute_shap_values, load_data_and_model

        config = get_config()
        data = load_data_and_model(config)

        shap_values = compute_shap_values(
            model=data["rf_model"],
            X_background=data["X_train"][:50],  # Small subset for speed
            X_explain=data["X_test"][:10],
            feature_names=data["feature_names"],
            max_samples=10,
        )

        assert shap_values.values.shape[0] == 10
        assert shap_values.values.shape[1] == len(data["feature_names"])


# ---------------------------------------------------------------------------
# Integration test: pipeline outputs
# ---------------------------------------------------------------------------

class TestExplainabilityOutputs:

    def test_outputs_exist_after_pipeline(self):
        """After running the pipeline, key output files should exist."""
        config = get_config()
        output_dir = PROJECT_ROOT / config["model"]["output_dir"] / "explainability"

        # These tests check for outputs created by run_explainability
        # Skip if pipeline hasn't been run yet
        if not output_dir.exists():
            pytest.skip("Explainability pipeline not yet run")

        expected_files = [
            "feature_importance.csv",
            "feature_importance.png",
            "shap_values.csv",
            "shap_summary_beeswarm.png",
            "shap_summary_bar.png",
            "interpretation_guide.md",
        ]

        for fname in expected_files:
            path = output_dir / fname
            assert path.exists(), f"Missing: {fname}"
            assert path.stat().st_size > 0, f"Empty: {fname}"
