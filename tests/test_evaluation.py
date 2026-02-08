"""
Tests for Phase 7 — Evaluation & Reporting.

Covers:
  - Data loading and model predictions
  - Plot generation (confusion matrix, ROC, PR, calibration)
  - Stress tests produce results
  - Per-generator analysis
  - Final report generation
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.utils.config import get_config, PROJECT_ROOT
from src.features.extractor import FeatureExtractor


# ─────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def eval_data():
    """Load data and models once for all tests in this module."""
    from src.evaluation.evaluate import load_data_and_models
    config = get_config()
    return load_data_and_models(config)


@pytest.fixture(scope="module")
def predictions(eval_data):
    """Run predictions once for all tests."""
    from src.evaluation.evaluate import predict_all
    return predict_all(eval_data)


# ─────────────────────────────────────────────────────
# Data & Model Loading
# ─────────────────────────────────────────────────────

class TestDataLoading:

    def test_models_loaded(self, eval_data):
        """All 4 models should be loaded."""
        assert len(eval_data["models"]) == 4
        expected = {"logistic_regression", "random_forest", "xgboost", "stacking_ensemble"}
        assert set(eval_data["models"].keys()) == expected

    def test_test_set_size(self, eval_data):
        """Test set should have 150 samples."""
        assert len(eval_data["y_test"]) == 150

    def test_predictions_shape(self, predictions, eval_data):
        """Each model should produce predictions for all test samples."""
        n = len(eval_data["y_test"])
        for name, pred in predictions.items():
            assert pred["y_pred"].shape == (n,), f"{name} y_pred shape wrong"
            assert pred["y_prob"].shape == (n,), f"{name} y_prob shape wrong"

    def test_predictions_valid(self, predictions):
        """Predictions should be binary, probabilities in [0, 1]."""
        for name, pred in predictions.items():
            assert set(np.unique(pred["y_pred"])).issubset({0, 1})
            assert np.all(pred["y_prob"] >= 0) and np.all(pred["y_prob"] <= 1)


# ─────────────────────────────────────────────────────
# Plot Generation
# ─────────────────────────────────────────────────────

class TestPlotGeneration:

    @pytest.fixture(autouse=True)
    def setup_output_dir(self):
        config = get_config()
        self.output_dir = PROJECT_ROOT / config["model"]["output_dir"] / "evaluation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_confusion_matrices(self, predictions, eval_data):
        from src.evaluation.evaluate import plot_confusion_matrices
        plot_confusion_matrices(predictions, eval_data["y_test"], self.output_dir)
        assert (self.output_dir / "confusion_matrices.png").exists()

    def test_roc_curves(self, predictions, eval_data):
        from src.evaluation.evaluate import plot_roc_curves
        plot_roc_curves(predictions, eval_data["y_test"], self.output_dir)
        assert (self.output_dir / "roc_curves.png").exists()

    def test_pr_curves(self, predictions, eval_data):
        from src.evaluation.evaluate import plot_pr_curves
        plot_pr_curves(predictions, eval_data["y_test"], self.output_dir)
        assert (self.output_dir / "precision_recall_curves.png").exists()

    def test_calibration_curves(self, predictions, eval_data):
        from src.evaluation.evaluate import plot_calibration_curves
        plot_calibration_curves(predictions, eval_data["y_test"], self.output_dir)
        assert (self.output_dir / "calibration_curves.png").exists()


# ─────────────────────────────────────────────────────
# Stress Tests
# ─────────────────────────────────────────────────────

class TestStressTests:

    def test_stress_tests_run(self, eval_data):
        """Stress tests should produce results for all cases."""
        from src.evaluation.evaluate import run_stress_tests

        config = get_config()
        output_dir = PROJECT_ROOT / config["model"]["output_dir"] / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = run_stress_tests(config, eval_data, output_dir)

        assert "mersenne_twister" in results
        assert "constant_bytes" in results
        assert "strong_64bit" in results
        assert "strong_256bit" in results
        assert "small_batch_10" in results
        assert "small_batch_50" in results

    def test_constant_bytes_classified_weak(self, eval_data):
        """Constant bytes should be classified as WEAK."""
        from src.evaluation.evaluate import run_stress_tests

        config = get_config()
        output_dir = PROJECT_ROOT / config["model"]["output_dir"] / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = run_stress_tests(config, eval_data, output_dir)
        assert results["constant_bytes"]["predicted"] == 0, \
            "Constant byte output should be WEAK"


# ─────────────────────────────────────────────────────
# Per-Generator Analysis
# ─────────────────────────────────────────────────────

class TestPerGeneratorAnalysis:

    def test_per_generator_returns_dataframe(self, predictions, eval_data):
        from src.evaluation.evaluate import per_generator_analysis

        config = get_config()
        output_dir = PROJECT_ROOT / config["model"]["output_dir"] / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = per_generator_analysis(
            predictions, eval_data["y_test"],
            eval_data["gen_test"], output_dir
        )
        assert isinstance(df, pd.DataFrame)
        assert "generator" in df.columns
        assert "accuracy" in df.columns
        assert len(df) > 0


# ─────────────────────────────────────────────────────
# Integration: Full Pipeline Outputs
# ─────────────────────────────────────────────────────

class TestEvaluationOutputs:

    def test_outputs_exist_after_pipeline(self):
        """After running evaluate(), key output files should exist."""
        config = get_config()
        output_dir = PROJECT_ROOT / config["model"]["output_dir"] / "evaluation"

        if not output_dir.exists():
            pytest.skip("Evaluation pipeline not yet run")

        expected = [
            "confusion_matrices.png",
            "roc_curves.png",
            "precision_recall_curves.png",
            "calibration_curves.png",
            "stress_test_results.json",
            "per_generator_accuracy.csv",
            "final_report.md",
        ]

        for fname in expected:
            path = output_dir / fname
            assert path.exists(), f"Missing: {fname}"
            assert path.stat().st_size > 0, f"Empty: {fname}"
