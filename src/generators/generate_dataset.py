"""
Dataset generation script for RandFusion.

Generates labeled batches of tokens from strong and weak generators,
validates them, and saves to data/raw/ as .npz files with metadata.

Usage:
    python -m src.generators.generate_dataset
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.utils.config import get_config, PROJECT_ROOT
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed
from src.generators.strong import STRONG_GENERATORS
from src.generators.weak import WEAK_GENERATORS

logger = get_logger(__name__)


def validate_batch(batch: np.ndarray, token_length_bits: int, batch_size: int) -> bool:
    """Check that a batch is valid and not degenerate.

    Catches: wrong shape, non-binary values, all-zeros, all-ones,
    or extremely low variance (constant tokens).

    Args:
        batch: Token batch array (batch_size, token_length_bits).
        token_length_bits: Expected number of bits per token.
        batch_size: Expected number of tokens.

    Returns:
        True if the batch passes all checks.
    """
    # Shape check
    if batch.shape != (batch_size, token_length_bits):
        logger.warning(f"Bad shape: {batch.shape}, expected ({batch_size}, {token_length_bits})")
        return False

    # Binary check (only 0s and 1s)
    if not np.all((batch == 0) | (batch == 1)):
        logger.warning("Batch contains non-binary values")
        return False

    # All-zeros or all-ones check
    total_ones = batch.sum()
    total_bits = batch_size * token_length_bits
    if total_ones == 0 or total_ones == total_bits:
        logger.warning("Degenerate batch: all zeros or all ones")
        return False

    # Very low variance check (all tokens identical)
    if np.all(batch == batch[0]):
        logger.warning("Degenerate batch: all tokens are identical")
        return False

    return True


def generate_dataset():
    """Generate the full labeled dataset and save to disk."""
    config = get_config()
    seed = set_global_seed()

    token_length_bits = config["token"]["length_bits"]
    batch_size = config["token"]["batch_size"]
    num_strong = config["dataset"]["num_strong_batches"]
    num_weak = config["dataset"]["num_weak_batches"]
    output_dir = PROJECT_ROOT / config["dataset"]["output_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating dataset: {num_strong} strong + {num_weak} weak batches")
    logger.info(f"Each batch: {batch_size} tokens × {token_length_bits} bits")
    logger.info(f"Output directory: {output_dir}")

    all_batches = []   # List of 2D arrays
    all_labels = []    # 1 = strong, 0 = weak
    all_metadata = []  # generator name for each batch

    # --- Strong batches ---
    strong_gen_names = list(STRONG_GENERATORS.keys())
    batches_per_strong_gen = num_strong // len(strong_gen_names)
    remainder = num_strong % len(strong_gen_names)

    logger.info(f"Strong generators: {strong_gen_names}")

    for idx, gen_name in enumerate(strong_gen_names):
        gen_func = STRONG_GENERATORS[gen_name]
        count = batches_per_strong_gen + (1 if idx < remainder else 0)

        logger.info(f"  Generating {count} batches from '{gen_name}'...")
        for _ in tqdm(range(count), desc=f"  {gen_name}", leave=False):
            batch = gen_func(batch_size, token_length_bits)
            if validate_batch(batch, token_length_bits, batch_size):
                all_batches.append(batch)
                all_labels.append(1)
                all_metadata.append(gen_name)
            else:
                logger.warning(f"  Skipping invalid batch from '{gen_name}'")

    # --- Weak batches ---
    weak_gen_names = list(WEAK_GENERATORS.keys())
    batches_per_weak_gen = num_weak // len(weak_gen_names)
    remainder = num_weak % len(weak_gen_names)

    logger.info(f"Weak generators: {weak_gen_names}")

    for idx, gen_name in enumerate(weak_gen_names):
        gen_func, extra_kwargs = WEAK_GENERATORS[gen_name]
        count = batches_per_weak_gen + (1 if idx < remainder else 0)

        logger.info(f"  Generating {count} batches from '{gen_name}'...")
        for batch_idx in tqdm(range(count), desc=f"  {gen_name}", leave=False):
            # Vary the seed per batch to get different samples
            kwargs = {**extra_kwargs, "seed": seed + batch_idx + idx * 10000}
            # Some generators have 'seed' as a positional-style kwarg,
            # but predictable_seed uses 'base_seed' differently
            if gen_name == "predictable_seed":
                kwargs = {"base_seed": seed + batch_idx * batch_size}
            elif gen_name.startswith("lcg"):
                kwargs = {"seed": seed + batch_idx * 7 + idx * 10000}
            batch = gen_func(batch_size, token_length_bits, **kwargs)
            if validate_batch(batch, token_length_bits, batch_size):
                all_batches.append(batch)
                all_labels.append(0)
                all_metadata.append(gen_name)
            else:
                logger.warning(f"  Skipping invalid batch from '{gen_name}'")

    # --- Save ---
    labels = np.array(all_labels, dtype=np.int8)
    metadata = all_metadata

    # Save batches as a single .npz file
    # batches: 3D array (num_samples, batch_size, token_length_bits)
    batches_array = np.array(all_batches, dtype=np.uint8)

    npz_path = output_dir / "dataset.npz"
    np.savez_compressed(
        npz_path,
        batches=batches_array,
        labels=labels,
    )

    # Save metadata as JSON
    meta_path = output_dir / "metadata.json"
    meta_info = {
        "total_samples": len(labels),
        "num_strong": int((labels == 1).sum()),
        "num_weak": int((labels == 0).sum()),
        "token_length_bits": token_length_bits,
        "batch_size": batch_size,
        "random_seed": seed,
        "generator_names": metadata,
        "strong_generators": list(STRONG_GENERATORS.keys()),
        "weak_generators": list(WEAK_GENERATORS.keys()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=2)

    logger.info(f"Dataset saved: {npz_path} ({batches_array.nbytes / 1e6:.1f} MB)")
    logger.info(f"Metadata saved: {meta_path}")
    logger.info(f"Total samples: {len(labels)} (strong={int((labels==1).sum())}, weak={int((labels==0).sum())})")

    return npz_path, meta_path


if __name__ == "__main__":
    generate_dataset()
