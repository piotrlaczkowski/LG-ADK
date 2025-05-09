"""Logging utility for LG-ADK using loguru.

Provides a get_logger function for consistent logging across the codebase.
"""
from loguru import logger


def get_logger(_name: str = None) -> logger.__class__:
    """Get the loguru logger instance (compatible signature).

    Args:
        _name (str, optional): Logger name (ignored, for compatibility).

    Returns:
        loguru.Logger: The loguru logger instance for application-wide logging.
    """
    return logger
