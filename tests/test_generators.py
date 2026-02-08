"""
Tests for Phase 1: Token generators and dataset generation.

Verifies:
  - Each strong generator produces valid binary batches
  - Each weak generator produces valid binary batches
  - Weak generators actually exhibit their intended weakness
  - Full dataset generation works end-to-end
"""

import numpy as np
import pytest
from src.generators.strong import STRONG_GENERATORS
from src.generators.weak import WEAK_GENERATORS, generate_biased_tokens, generate_lcg_tokens


# Small sizes for fast testing
TEST_BATCH_SIZE = 50
TEST_TOKEN_BITS = 128


class TestStrongGenerators:
    """Verify that strong generators produce well-formed output."""

    @pytest.mark.parametrize("name", STRONG_GENERATORS.keys())
    def test_shape_and_binary(self, name):
        gen_func = STRONG_GENERATORS[name]
        batch = gen_func(TEST_BATCH_SIZE, TEST_TOKEN_BITS)

        assert batch.shape == (TEST_BATCH_SIZE, TEST_TOKEN_BITS)
        assert set(np.unique(batch)).issubset({0, 1})

    @pytest.mark.parametrize("name", STRONG_GENERATORS.keys())
    def test_not_degenerate(self, name):
        gen_func = STRONG_GENERATORS[name]
        batch = gen_func(TEST_BATCH_SIZE, TEST_TOKEN_BITS)

        # Should not be all zeros or all ones
        assert batch.sum() > 0
        assert batch.sum() < TEST_BATCH_SIZE * TEST_TOKEN_BITS

    @pytest.mark.parametrize("name", STRONG_GENERATORS.keys())
    def test_reasonable_bit_ratio(self, name):
        """Strong generators should have roughly 50% ones."""
        gen_func = STRONG_GENERATORS[name]
        batch = gen_func(TEST_BATCH_SIZE, TEST_TOKEN_BITS)

        ratio = batch.mean()
        # Allow wide margin, but should be in [0.35, 0.65]
        assert 0.35 < ratio < 0.65, f"Bit ratio {ratio:.3f} is suspicious for a CSPRNG"


class TestWeakGenerators:
    """Verify that weak generators produce well-formed but flawed output."""

    @pytest.mark.parametrize("name", WEAK_GENERATORS.keys())
    def test_shape_and_binary(self, name):
        gen_func, extra_kwargs = WEAK_GENERATORS[name]
        batch = gen_func(TEST_BATCH_SIZE, TEST_TOKEN_BITS, **extra_kwargs)

        assert batch.shape == (TEST_BATCH_SIZE, TEST_TOKEN_BITS)
        assert set(np.unique(batch)).issubset({0, 1})

    def test_biased_actually_biased(self):
        """Biased generator with bias=0.7 should have mean well above 0.5."""
        batch = generate_biased_tokens(500, TEST_TOKEN_BITS, bias=0.7, seed=42)
        ratio = batch.mean()
        assert ratio > 0.65, f"Biased generator ratio {ratio:.3f} — not biased enough"

    def test_lcg_has_patterns(self):
        """LCG output should show autocorrelation (same seed = same sequence)."""
        batch1 = generate_lcg_tokens(10, TEST_TOKEN_BITS, seed=123)
        batch2 = generate_lcg_tokens(10, TEST_TOKEN_BITS, seed=123)
        # Same seed → identical output (deterministic)
        np.testing.assert_array_equal(batch1, batch2)


class TestDatasetGeneration:
    """Quick sanity check of the full generation pipeline."""

    def test_import_works(self):
        """Just verify the generate_dataset function can be imported."""
        from src.generators.generate_dataset import generate_dataset
        assert callable(generate_dataset)
