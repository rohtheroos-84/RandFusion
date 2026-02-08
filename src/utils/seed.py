"""
Seed management for reproducibility.

Sets random seeds for Python stdlib, NumPy, and any other
libraries that need deterministic behaviour.

Usage:
    from src.utils.seed import set_global_seed
    set_global_seed()           # uses config value
    set_global_seed(123)        # explicit override
"""

import random
import numpy as np
from src.utils.config import get_config


def set_global_seed(seed: int | None = None) -> int:
    """Set random seeds across all relevant libraries.

    Args:
        seed: Explicit seed value. If None, reads from config.yaml.

    Returns:
        The seed that was actually applied.
    """
    if seed is None:
        config = get_config()
        seed = config.get("random_seed", 42)

    random.seed(seed)
    np.random.seed(seed)

    return seed
