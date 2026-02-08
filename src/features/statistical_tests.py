"""
Non-NIST statistical randomness tests for RandFusion.

These complement the NIST tests with information-theoretic and
structural measures. Each takes a 1D bit array (the concatenated
bitstream for one batch) and returns numerical features.

Tests:
  - Shannon entropy (per-byte)
  - Min-entropy estimation
  - Run-length statistics
  - Autocorrelation at multiple lags
  - Compression ratio (zlib)
"""

import zlib
import numpy as np
from collections import Counter


def shannon_entropy(bits: np.ndarray, block_size: int = 8) -> dict:
    """Shannon entropy computed on non-overlapping byte-sized blocks.

    For a truly random sequence of bytes, Shannon entropy should be
    close to 8.0 bits (maximum for 256 possible values).

    Args:
        bits: 1D array of 0/1 values.
        block_size: Number of bits per block (default 8 = one byte).

    Returns:
        Dict with entropy value (0 to block_size).
    """
    n = len(bits)
    num_blocks = n // block_size
    if num_blocks == 0:
        return {"shannon_entropy": 0.0}

    # Convert bit blocks to integer values
    blocks = bits[:num_blocks * block_size].reshape(num_blocks, block_size)
    # Pack bits to integers
    powers = 2 ** np.arange(block_size - 1, -1, -1)
    values = blocks @ powers

    # Count frequencies
    counts = Counter(values)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])

    # Shannon entropy
    entropy = -np.sum(probs * np.log2(probs))

    return {"shannon_entropy": float(entropy)}


def min_entropy(bits: np.ndarray, block_size: int = 8) -> dict:
    """Min-entropy estimation using most-common-value estimator.

    Min-entropy = -log2(max probability). For truly random bytes,
    max probability ≈ 1/256, so min-entropy ≈ 8.0.

    Lower min-entropy means more predictable output.

    Args:
        bits: 1D array of 0/1 values.
        block_size: Number of bits per block.

    Returns:
        Dict with min-entropy and the max observed probability.
    """
    n = len(bits)
    num_blocks = n // block_size
    if num_blocks == 0:
        return {"min_entropy": 0.0, "max_probability": 1.0}

    blocks = bits[:num_blocks * block_size].reshape(num_blocks, block_size)
    powers = 2 ** np.arange(block_size - 1, -1, -1)
    values = blocks @ powers

    counts = Counter(values)
    max_count = max(counts.values())
    max_prob = max_count / num_blocks

    # Avoid log(0)
    if max_prob >= 1.0:
        h_min = 0.0
    else:
        h_min = -np.log2(max_prob)

    return {"min_entropy": float(h_min), "max_probability": float(max_prob)}


def run_length_statistics(bits: np.ndarray) -> dict:
    """Run-length statistics of the bitstream.

    A "run" is a maximal sequence of identical consecutive bits.
    For a random sequence, run lengths should follow a geometric distribution.

    Returns mean, standard deviation, max run length, and the number of runs.
    """
    if len(bits) == 0:
        return {"run_mean": 0.0, "run_std": 0.0, "run_max": 0, "num_runs": 0}

    # Find run boundaries
    changes = np.where(bits[1:] != bits[:-1])[0]
    run_starts = np.concatenate([[0], changes + 1])
    run_ends = np.concatenate([changes + 1, [len(bits)]])
    run_lengths = run_ends - run_starts

    return {
        "run_mean": float(run_lengths.mean()),
        "run_std": float(run_lengths.std()),
        "run_max": int(run_lengths.max()),
        "num_runs": int(len(run_lengths)),
    }


def autocorrelation(bits: np.ndarray, lags: list[int] = None) -> dict:
    """Autocorrelation of the bitstream at specified lags.

    Measures how correlated a bit is with the bit `lag` positions later.
    For truly random bits, autocorrelation should be near 0 for all lags.

    High autocorrelation at lag d means the sequence repeats with period d.

    Args:
        bits: 1D array of 0/1 values.
        lags: List of lag values to check. Defaults to [1, 2, 4, 8, 16].

    Returns:
        Dict with autocorrelation value for each lag.
    """
    if lags is None:
        lags = [1, 2, 4, 8, 16]

    n = len(bits)
    # Map to ±1 for correlation
    x = 2 * bits.astype(np.float64) - 1

    results = {}
    for lag in lags:
        if lag >= n:
            results[f"autocorr_lag_{lag}"] = 0.0
            continue

        # Autocorrelation at this lag
        corr = np.sum(x[:n - lag] * x[lag:]) / (n - lag)
        results[f"autocorr_lag_{lag}"] = float(corr)

    return results


def compression_ratio(bits: np.ndarray, level: int = 6) -> dict:
    """Compression ratio using zlib.

    Truly random data is incompressible — the compressed size should be
    close to (or slightly larger than) the original. If the data compresses
    significantly, it has structure/patterns.

    Returns ratio = compressed_size / original_size.
    Values near 1.0 = good (incompressible), much less than 1.0 = bad.

    Args:
        bits: 1D array of 0/1 values.
        level: zlib compression level (1-9).

    Returns:
        Dict with compression ratio and sizes.
    """
    # Pack bits to bytes for realistic compression
    num_bytes = len(bits) // 8
    bit_array = bits[:num_bytes * 8].reshape(num_bytes, 8)
    powers = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    raw_bytes = (bit_array @ powers).astype(np.uint8).tobytes()

    compressed = zlib.compress(raw_bytes, level)
    original_size = len(raw_bytes)
    compressed_size = len(compressed)
    ratio = compressed_size / original_size if original_size > 0 else 1.0

    return {
        "compression_ratio": float(ratio),
        "original_bytes": int(original_size),
        "compressed_bytes": int(compressed_size),
    }
