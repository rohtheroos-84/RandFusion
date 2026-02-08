"""
Strong randomness generators for RandFusion.

These produce cryptographically secure random tokens using
OS-level CSPRNGs. They serve as the "good randomness" class
in the training dataset (label = 1).
"""

import os
import secrets
import numpy as np


def generate_secrets_tokens(batch_size: int, token_length_bits: int) -> np.ndarray:
    """Generate tokens using Python's `secrets` module (CSPRNG).

    The secrets module uses the OS's best available randomness source.
    This is what production systems should use for token generation.

    Args:
        batch_size: Number of tokens to generate.
        token_length_bits: Length of each token in bits.

    Returns:
        2D numpy array of shape (batch_size, token_length_bits) with 0/1 values.
    """
    token_length_bytes = token_length_bits // 8
    batch = np.zeros((batch_size, token_length_bits), dtype=np.uint8)

    for i in range(batch_size):
        raw_bytes = secrets.token_bytes(token_length_bytes)
        bits = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))
        batch[i] = bits[:token_length_bits]

    return batch


def generate_urandom_tokens(batch_size: int, token_length_bits: int) -> np.ndarray:
    """Generate tokens using os.urandom (kernel-level RNG).

    os.urandom reads directly from the OS entropy pool (/dev/urandom on Linux,
    CryptGenRandom on Windows). Virtually identical quality to secrets.

    Args:
        batch_size: Number of tokens to generate.
        token_length_bits: Length of each token in bits.

    Returns:
        2D numpy array of shape (batch_size, token_length_bits) with 0/1 values.
    """
    token_length_bytes = token_length_bits // 8
    batch = np.zeros((batch_size, token_length_bits), dtype=np.uint8)

    for i in range(batch_size):
        raw_bytes = os.urandom(token_length_bytes)
        bits = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8))
        batch[i] = bits[:token_length_bits]

    return batch


# Registry: name -> generator function
STRONG_GENERATORS = {
    "secrets_csprng": generate_secrets_tokens,
    "os_urandom": generate_urandom_tokens,
}
