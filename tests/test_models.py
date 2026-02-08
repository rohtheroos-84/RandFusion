"""
Tests for Phase 4: ML model training and ensemble.

Verifies:
  - All model artifacts are saved to disk  
  - Models can be loaded and produce predictions
  - Results JSON contains expected keys
  - Ensemble outperforms random guessing
  - Predictions are valid probabilities
"""

import json
import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.utils.config import get_config, PROJECT_ROOT
from src.features.extractor import FeatureExtractor


@pytest.fixture
def models_dir():
    config = get_config()
    return PROJECT_ROOT / config["model"]["output_dir"]


@pytest.fixture
def results(models_dir):
    with open(models_dir / "results.json") as f:
        return json.load(f)


@pytest.fixture
def test_data():
    config = get_config()
    csv_path = PROJECT_ROOT / config["dataset"]["processed_dir"] / "features.csv"
    df = pd.read_csv(csv_path)
    fe = FeatureExtractor(config)
    feature_names = fe.feature_names()
    return df[feature_names].values[:10], df["label"].values[:10]


class TestModelArtifacts:

    def test_scaler_saved(self, models_dir):
        assert (models_dir / "scaler.joblib").exists()

    def test_base_models_saved(self, models_dir):
        for name in ["logistic_regression", "random_forest", "xgboost"]:
            assert (models_dir / f"{name}.joblib").exists()

    def test_ensemble_saved(self, models_dir):
        assert (models_dir / "stacking_ensemble.joblib").exists()

    def test_results_saved(self, models_dir):
        assert (models_dir / "results.json").exists()


class TestModelLoading:

    def test_load_and_predict_ensemble(self, models_dir, test_data):
        model = joblib.load(models_dir / "stacking_ensemble.joblib")
        X, _ = test_data
        preds = model.predict(X)
        probs = model.predict_proba(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})
        assert probs.shape == (len(X), 2)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_load_and_predict_rf(self, models_dir, test_data):
        model = joblib.load(models_dir / "random_forest.joblib")
        X, _ = test_data
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})


class TestResults:

    def test_results_has_all_models(self, results):
        expected = ["logistic_regression", "random_forest", "xgboost", "stacking_ensemble"]
        for name in expected:
            assert name in results["test_results"], f"Missing {name}"

    def test_metrics_present(self, results):
        for name, metrics in results["test_results"].items():
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                assert key in metrics, f"Missing {key} for {name}"

    def test_above_random_chance(self, results):
        """All models should beat 50% accuracy (random guessing)."""
        for name, metrics in results["test_results"].items():
            assert metrics["accuracy"] > 0.5, f"{name} accuracy below chance"
            assert metrics["roc_auc"] > 0.5, f"{name} AUC below chance"

    def test_split_sizes(self, results):
        sizes = results["split_sizes"]
        assert sizes["train"] == 700
        assert sizes["val"] == 150
        assert sizes["test"] == 150
