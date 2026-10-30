"""Simple, consistent logging for all LoomIQ modules.

Example:
    >>> from loomiq.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Training started")
"""

import logging
import os

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Return a logger that prints to the console.

    Args:
        name: Usually ``__name__`` of the calling module.
        level: Log level such as ``"INFO"``. Defaults to the ``LOG_LEVEL``
            environment variable, or ``"INFO"`` if that is not set.

    Returns:
        A configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level or os.getenv("LOG_LEVEL", "INFO"))

    # Avoid adding duplicate handlers when the logger is requested again
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
