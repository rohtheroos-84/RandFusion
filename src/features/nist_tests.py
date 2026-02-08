"""
NIST SP 800-22 statistical randomness tests (subset).

Implements the tests that work well on the bitstream sizes we produce
(128,000 bits = 1000 tokens × 128 bits). Each function takes a 1D bit
array and returns a p-value (and optionally a test statistic).

Higher p-value → more consistent with true randomness.
p-value < 0.01 → evidence of non-randomness.

Reference: https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final

Tests included:
  1. Frequency (monobit)
  2. Block frequency
  3. Runs
  4. Longest run of ones in a block
  5. Serial (two patterns)
  6. Approximate entropy
  7. Cumulative sums (forward and backward)
"""

import numpy as np
from scipy.special import erfc, gammaincc
from scipy.stats import chi2


def frequency_test(bits: np.ndarray) -> dict:
    """NIST Test 1: Frequency (Monobit) Test.

    Checks whether the proportion of 0s and 1s in the entire sequence
    is approximately equal, as expected for a truly random sequence.

    The test statistic is |sum of ±1 mapped bits| / sqrt(n).
    """
    n = len(bits)
    # Map 0→-1, 1→+1
    s = np.sum(2 * bits.astype(np.float64) - 1)
    s_obs = abs(s) / np.sqrt(n)
    p_value = erfc(s_obs / np.sqrt(2))
    return {"p_value": float(p_value), "statistic": float(s_obs)}


def block_frequency_test(bits: np.ndarray, block_size: int = 128) -> dict:
    """NIST Test 2: Frequency Within a Block Test.

    Divides the sequence into non-overlapping blocks and checks whether
    the proportion of ones in each block is approximately 0.5.
    """
    n = len(bits)
    num_blocks = n // block_size

    if num_blocks == 0:
        return {"p_value": 0.0, "statistic": float("inf")}

    # Proportion of ones in each block
    blocks = bits[:num_blocks * block_size].reshape(num_blocks, block_size)
    proportions = blocks.mean(axis=1)

    # Chi-squared statistic
    chi_sq = 4 * block_size * np.sum((proportions - 0.5) ** 2)
    p_value = gammaincc(num_blocks / 2, chi_sq / 2)

    return {"p_value": float(p_value), "statistic": float(chi_sq)}


def runs_test(bits: np.ndarray) -> dict:
    """NIST Test 3: Runs Test.

    Checks whether the number of runs (uninterrupted sequences of identical
    bits) is consistent with a random sequence. Requires that the frequency
    test is not badly failed first (pre-test on proportion).
    """
    n = len(bits)
    pi = bits.mean()

    # Pre-test: if proportion is too far from 0.5, test is not applicable
    tau = 2.0 / np.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return {"p_value": 0.0, "statistic": float("nan")}

    # Count runs: number of positions where bit changes
    runs = 1 + np.sum(bits[1:] != bits[:-1])

    # Expected and variance
    p_value = erfc(abs(runs - 2 * n * pi * (1 - pi)) /
                   (2 * np.sqrt(2 * n) * pi * (1 - pi)))

    return {"p_value": float(p_value), "statistic": float(runs)}


def longest_run_of_ones_test(bits: np.ndarray) -> dict:
    """NIST Test 4: Longest Run of Ones in a Block.

    Checks whether the longest run of ones within blocks is consistent
    with what's expected for a random sequence. Block size and expected
    values depend on the sequence length.
    """
    n = len(bits)

    # Choose parameters based on sequence length
    if n < 6272:
        block_size = 8
        k = 3
        expected = np.array([0.2148, 0.3672, 0.2305, 0.1875])
        boundaries = [1, 2, 3, 4]  # categories: <=1, 2, 3, >=4
    elif n < 750000:
        block_size = 128
        k = 5
        expected = np.array([0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124])
        boundaries = [4, 5, 6, 7, 8, 9]  # <=4, 5, 6, 7, 8, >=9
    else:
        block_size = 10000
        k = 6
        expected = np.array([0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727])
        boundaries = [10, 11, 12, 13, 14, 15, 16]

    num_blocks = n // block_size
    if num_blocks == 0:
        return {"p_value": 0.0, "statistic": float("inf")}

    # Find longest run in each block
    longest_runs = np.zeros(num_blocks, dtype=int)
    for i in range(num_blocks):
        block = bits[i * block_size:(i + 1) * block_size]
        max_run = 0
        current_run = 0
        for bit in block:
            if bit == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        longest_runs[i] = max_run

    # Count frequencies in each category
    num_categories = len(expected)
    observed = np.zeros(num_categories)

    for run_len in longest_runs:
        if run_len <= boundaries[0]:
            observed[0] += 1
        elif run_len >= boundaries[-1]:
            observed[-1] += 1
        else:
            for j in range(1, num_categories - 1):
                if run_len == boundaries[j]:
                    observed[j] += 1
                    break

    # Chi-squared
    chi_sq = np.sum((observed - num_blocks * expected) ** 2 / (num_blocks * expected))
    p_value = gammaincc(k / 2, chi_sq / 2)

    return {"p_value": float(p_value), "statistic": float(chi_sq)}


def serial_test(bits: np.ndarray, block_length: int = 3) -> dict:
    """NIST Test 11: Serial Test.

    Tests the uniformity of overlapping m-bit patterns. Produces two
    p-values that check for uniformity of pattern frequencies.
    """
    n = len(bits)
    m = block_length

    # Augment sequence (wrap around)
    augmented = np.concatenate([bits, bits[:m - 1]])

    def count_patterns(seq, pattern_len):
        """Count frequency of each m-bit pattern."""
        num_patterns = 2 ** pattern_len
        counts = np.zeros(num_patterns)
        for i in range(n):
            pattern = 0
            for j in range(pattern_len):
                pattern = (pattern << 1) | int(seq[i + j])
            counts[pattern] += 1
        return counts

    def psi_sq(pattern_len):
        if pattern_len == 0:
            return 0.0
        counts = count_patterns(augmented, pattern_len)
        return (2 ** pattern_len / n) * np.sum(counts ** 2) - n

    psi_m = psi_sq(m)
    psi_m1 = psi_sq(m - 1)
    psi_m2 = psi_sq(m - 2) if m >= 2 else 0.0

    delta1 = psi_m - psi_m1
    delta2 = psi_m - 2 * psi_m1 + psi_m2

    p_value1 = gammaincc(2 ** (m - 2), delta1 / 2)
    p_value2 = gammaincc(2 ** (m - 3), delta2 / 2) if m >= 3 else 1.0

    return {
        "p_value_1": float(p_value1),
        "p_value_2": float(p_value2),
        "statistic_delta1": float(delta1),
        "statistic_delta2": float(delta2),
    }


def approximate_entropy_test(bits: np.ndarray, block_length: int = 3) -> dict:
    """NIST Test 12: Approximate Entropy Test.

    Compares the frequency of overlapping blocks of two consecutive lengths
    (m and m+1). Non-random sequences show less entropy difference.
    """
    n = len(bits)
    m = block_length

    def phi(pattern_len):
        augmented = np.concatenate([bits, bits[:pattern_len - 1]])
        num_patterns = 2 ** pattern_len
        counts = np.zeros(num_patterns)
        for i in range(n):
            pattern = 0
            for j in range(pattern_len):
                pattern = (pattern << 1) | int(augmented[i + j])
            counts[pattern] += 1
        # Normalize
        probs = counts / n
        # Log sum (handle zeros)
        probs = probs[probs > 0]
        return np.sum(probs * np.log(probs))

    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    apen = phi_m - phi_m1
    chi_sq = 2 * n * (np.log(2) - apen)
    p_value = gammaincc(2 ** (m - 1), chi_sq / 2)

    return {"p_value": float(p_value), "statistic": float(chi_sq), "apen": float(apen)}


def cumulative_sums_test(bits: np.ndarray) -> dict:
    """NIST Test 13: Cumulative Sums Test.

    Checks whether the cumulative sum of the adjusted (-1, +1) sequence
    is too large, indicating non-randomness. Tests both forward and backward.
    """
    n = len(bits)
    adjusted = 2 * bits.astype(np.float64) - 1

    results = {}
    for mode, seq in [("forward", adjusted), ("backward", adjusted[::-1])]:
        cumsum = np.cumsum(seq)
        z = np.max(np.abs(cumsum))

        # Compute p-value using the formula from NIST
        sum1 = 0.0
        start = int((-n / z + 1) / 4)
        end = int((n / z - 1) / 4) + 1
        for k in range(start, end + 1):
            from scipy.stats import norm
            sum1 += (norm.cdf((4 * k + 1) * z / np.sqrt(n)) -
                     norm.cdf((4 * k - 1) * z / np.sqrt(n)))

        sum2 = 0.0
        start2 = int((-n / z - 3) / 4)
        end2 = int((n / z - 1) / 4) + 1
        for k in range(start2, end2 + 1):
            sum2 += (norm.cdf((4 * k + 3) * z / np.sqrt(n)) -
                     norm.cdf((4 * k + 1) * z / np.sqrt(n)))

        p_value = 1.0 - sum1 + sum2
        p_value = max(0.0, min(1.0, p_value))  # clamp

        results[f"p_value_{mode}"] = float(p_value)
        results[f"statistic_{mode}"] = float(z)

    return results
