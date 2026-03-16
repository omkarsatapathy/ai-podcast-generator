"""Logging configuration."""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "podcast_generator", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger instance."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


# Default logger
logger = setup_logger()
