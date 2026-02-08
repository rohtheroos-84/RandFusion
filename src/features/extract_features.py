"""
Feature extraction pipeline for RandFusion.

Loads the raw token batches from Phase 1, runs the FeatureExtractor
on each batch, and saves the resulting feature matrix as a CSV.

Usage:
    python -m src.features.extract_features
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.features.extractor import FeatureExtractor

logger = get_logger(__name__)


def extract_features():
    """Run feature extraction on the full dataset and save as CSV."""
    config = get_config()
    set_global_seed()

    raw_dir = PROJECT_ROOT / config["dataset"]["output_dir"]
    processed_dir = PROJECT_ROOT / config["dataset"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load raw dataset
    logger.info(f"Loading raw dataset from {raw_dir / 'dataset.npz'}")
    data = np.load(raw_dir / "dataset.npz")
    batches = data["batches"]   # (num_samples, batch_size, token_bits)
    labels = data["labels"]     # (num_samples,)

    with open(raw_dir / "metadata.json") as f:
        metadata = json.load(f)
    generator_names = metadata["generator_names"]

    num_samples = len(labels)
    logger.info(f"Loaded {num_samples} batches ({(labels==1).sum()} strong, {(labels==0).sum()} weak)")

    # Initialize extractor
    fe = FeatureExtractor(config)
    feature_names = fe.feature_names()
    logger.info(f"Extracting {fe.num_features()} features per batch...")

    # Extract features for each batch
    rows = []
    for i in tqdm(range(num_samples), desc="Extracting features"):
        batch = batches[i]  # (batch_size, token_bits)
        features = fe.extract(batch)
        rows.append(features)

    # Build DataFrame
    df = pd.DataFrame(rows, columns=feature_names)
    df["label"] = labels
    df["generator"] = generator_names

    # Sanity checks
    nan_count = df[feature_names].isna().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in features — filling with 0")
        df[feature_names] = df[feature_names].fillna(0.0)

    inf_count = np.isinf(df[feature_names].values).sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} Inf values in features — clipping")
        df[feature_names] = df[feature_names].replace([np.inf, -np.inf], 0.0)

    # Save
    csv_path = processed_dir / "features.csv"
    df.to_csv(csv_path, index=False)

    logger.info(f"Feature matrix saved: {csv_path}")
    logger.info(f"Shape: {df.shape[0]} samples × {df.shape[1]} columns ({len(feature_names)} features + label + generator)")

    # Print summary statistics
    logger.info("\n--- Feature Summary (mean ± std) ---")
    for col in feature_names:
        strong_vals = df.loc[df["label"] == 1, col]
        weak_vals = df.loc[df["label"] == 0, col]
        logger.info(
            f"  {col:30s}  strong: {strong_vals.mean():8.4f} ± {strong_vals.std():6.4f}  |  "
            f"weak: {weak_vals.mean():8.4f} ± {weak_vals.std():6.4f}"
        )

    return csv_path


if __name__ == "__main__":
    extract_features()
