"""
Smoke test — verifies that the project skeleton works end-to-end.

Run with:  python -m tests.test_smoke
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed


def main():
    # 1. Load config
    config = get_config()
    print(f"[OK] Config loaded — token length: {config['token']['length_bits']} bits")

    # 2. Set seed
    seed = set_global_seed()
    print(f"[OK] Global seed set to {seed}")

    # 3. Logger
    logger = get_logger("smoke_test")
    logger.info("Logger is working.")

    # 4. Verify directories exist
    for d in ["data/raw", "data/processed", "models", "notebooks", "configs"]:
        p = Path(d)
        assert p.exists(), f"Directory missing: {d}"
    print("[OK] All expected directories exist")

    # 5. Quick numpy check
    import numpy as np
    arr = np.random.rand(10)
    print(f"[OK] NumPy working — sample array mean: {arr.mean():.4f}")

    print("\n✅ Phase 0 smoke test passed. Project skeleton is ready.")


if __name__ == "__main__":
    main()
