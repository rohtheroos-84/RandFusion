"""
Logging setup for RandFusion.

Provides a consistent logger that reads format/level from config.yaml.
Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting feature extraction...")
"""

import logging
from src.utils.config import get_config


def get_logger(name: str) -> logging.Logger:
    """Create and return a configured logger.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        Configured logging.Logger instance.
    """
    config = get_config()
    log_cfg = config.get("logging", {})

    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt = log_cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
