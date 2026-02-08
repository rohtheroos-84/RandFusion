"""
Weak randomness generators for RandFusion.

These produce tokens with intentional randomness flaws — biases, short
periods, predictable seeds, low entropy. They serve as the "weak randomness"
class in the training dataset (label = 0).

Each generator simulates a different real-world weakness:
  - LCG: short period, linear structure (common in old stdlib rand())
  - Biased coin: bits are not equally likely (e.g., 70% chance of 1)
  - Repeating seed: same short random block repeated (low effective entropy)
  - Predictable seed: time-seeded stdlib Random with sequential seeds
  - XOR collapse: XOR-ing CSPRNG output down to fewer effective bits
"""

import random
import struct
import numpy as np


def generate_lcg_tokens(
    batch_size: int, token_length_bits: int, seed: int = 12345
) -> np.ndarray:
    """Linear Congruential Generator with small modulus.

    LCG formula: state = (a * state + c) mod m
    With a small modulus (2^16), the period is at most 65536 — far too short
    for cryptographic use. The output is highly predictable.

    Args:
        batch_size: Number of tokens to generate.
        token_length_bits: Length of each token in bits.
        seed: Initial LCG state.

    Returns:
        2D numpy array of shape (batch_size, token_length_bits) with 0/1 values.
    """
    # Classic LCG parameters with small modulus
    a, c, m = 1103515245, 12345, 2**16
    state = seed & 0xFFFF

    token_length_bytes = token_length_bits // 8
    batch = np.zeros((batch_size, token_length_bits), dtype=np.uint8)

    for i in range(batch_size):
        raw = bytearray()
        for _ in range(token_length_bytes):
            state = (a * state + c) % m
            raw.append(state & 0xFF)
        bits = np.unpackbits(np.frombuffer(bytes(raw), dtype=np.uint8))
        batch[i] = bits[:token_length_bits]

    return batch


def generate_biased_tokens(
    batch_size: int, token_length_bits: int, bias: float = 0.7, seed: int = 42
) -> np.ndarray:
    """Biased coin — each bit has probability `bias` of being 1.

    Real randomness requires P(bit=0) = P(bit=1) = 0.5. This generator
    violates that, producing excess 1s (or 0s). A simple frequency test
    should catch this, but borderline biases (e.g., 0.52) might slip through.

    Args:
        batch_size: Number of tokens.
        token_length_bits: Bits per token.
        bias: Probability of a bit being 1 (0.5 = fair, >0.5 = biased toward 1).
        seed: Random seed for numpy.

    Returns:
        2D numpy array of shape (batch_size, token_length_bits) with 0/1 values.
    """
    rng = np.random.RandomState(seed)
    batch = (rng.random((batch_size, token_length_bits)) < bias).astype(np.uint8)
    return batch


def generate_repeating_seed_tokens(
    batch_size: int, token_length_bits: int, seed_length_bits: int = 32, seed: int = 99
) -> np.ndarray:
    """Repeating short random block — low effective entropy.

    Generates a small random seed block (e.g., 32 bits) and tiles it to fill
    each token. The effective entropy is only seed_length_bits regardless of
    the token length — a severe weakness.

    Args:
        batch_size: Number of tokens.
        token_length_bits: Bits per token.
        seed_length_bits: Length of the repeating block (much smaller than token).
        seed: Random seed.

    Returns:
        2D numpy array of shape (batch_size, token_length_bits) with 0/1 values.
    """
    rng = np.random.RandomState(seed)
    batch = np.zeros((batch_size, token_length_bits), dtype=np.uint8)

    for i in range(batch_size):
        # Generate a short random seed block
        seed_block = rng.randint(0, 2, size=seed_length_bits).astype(np.uint8)
        # Tile it to fill the token
        repeats_needed = (token_length_bits // seed_length_bits) + 1
        tiled = np.tile(seed_block, repeats_needed)
        batch[i] = tiled[:token_length_bits]

    return batch


def generate_predictable_seed_tokens(
    batch_size: int, token_length_bits: int, base_seed: int = 0
) -> np.ndarray:
    """Time-seeded stdlib Random with sequential predictable seeds.

    Simulates the anti-pattern of seeding random.Random(timestamp) where
    timestamps increment predictably. An attacker who knows the approximate
    time can brute-force the seed and regenerate all tokens.

    Args:
        batch_size: Number of tokens.
        token_length_bits: Bits per token.
        base_seed: Starting seed (incremented by 1 for each token).

    Returns:
        2D numpy array of shape (batch_size, token_length_bits) with 0/1 values.
    """
    token_length_bytes = token_length_bits // 8
    batch = np.zeros((batch_size, token_length_bits), dtype=np.uint8)

    for i in range(batch_size):
        rng = random.Random(base_seed + i)  # sequential, predictable seed
        raw = bytes([rng.randint(0, 255) for _ in range(token_length_bytes)])
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
        batch[i] = bits[:token_length_bits]

    return batch


def generate_xor_collapse_tokens(
    batch_size: int, token_length_bits: int, collapse_factor: int = 4, seed: int = 77
) -> np.ndarray:
    """XOR-collapsed output — reduces effective entropy.

    Generates token_length_bits * collapse_factor random bits, then XORs
    groups of `collapse_factor` bits together. The result looks somewhat
    random but has reduced entropy due to the XOR folding.

    Args:
        batch_size: Number of tokens.
        token_length_bits: Bits per token.
        collapse_factor: How many bits to XOR into one (higher = worse randomness).
        seed: Random seed.

    Returns:
        2D numpy array of shape (batch_size, token_length_bits) with 0/1 values.
    """
    rng = np.random.RandomState(seed)
    batch = np.zeros((batch_size, token_length_bits), dtype=np.uint8)

    for i in range(batch_size):
        # Generate extra bits
        expanded = rng.randint(0, 2, size=token_length_bits * collapse_factor).astype(np.uint8)
        # XOR groups of collapse_factor bits together
        reshaped = expanded.reshape(token_length_bits, collapse_factor)
        collapsed = np.bitwise_xor.reduce(reshaped, axis=1)
        batch[i] = collapsed

    return batch


# Registry: name -> (generator function, extra kwargs)
WEAK_GENERATORS = {
    "lcg_small_state": (generate_lcg_tokens, {}),
    "biased_070": (generate_biased_tokens, {"bias": 0.7}),
    "biased_055": (generate_biased_tokens, {"bias": 0.55}),
    "repeating_seed_32": (generate_repeating_seed_tokens, {"seed_length_bits": 32}),
    "repeating_seed_16": (generate_repeating_seed_tokens, {"seed_length_bits": 16}),
    "predictable_seed": (generate_predictable_seed_tokens, {}),
    "xor_collapse_4": (generate_xor_collapse_tokens, {"collapse_factor": 4}),
    "xor_collapse_8": (generate_xor_collapse_tokens, {"collapse_factor": 8}),
}
