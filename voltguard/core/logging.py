"""Structured logging for VoltGuard.

Replaces all print() calls with proper leveled, formatted logging.
Usage:
    from voltguard.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Model trained", extra={"accuracy": 0.96})
"""

from __future__ import annotations

import logging
import sys


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger("voltguard")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger. Automatically sets up logging on first call."""
    setup_logging()
    return logging.getLogger(name)
