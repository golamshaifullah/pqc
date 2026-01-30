"""Provide minimal logging helpers for CLI-style output."""

from __future__ import annotations
import logging

_LOGGER = logging.getLogger("pqc")


def configure_logging(*, level: str | int = "INFO", fmt: str = "%(message)s") -> None:
    """Configure PQC logging once.

    Args:
        level (str | int): Logging level (e.g., "INFO", "DEBUG").
        fmt (str): Logging format string.
    """
    if _LOGGER.handlers:
        return
    logging.basicConfig(level=level, format=fmt)


def warn(msg: str) -> None:
    """Print a warning message to stderr.

    Args:
        msg (str): Warning message text.

    Examples:
        >>> warn("Missing metadata")
    """
    configure_logging()
    _LOGGER.warning(msg)


def info(msg: str) -> None:
    """Print an informational message to stdout.

    Args:
        msg (str): Message text.

    Examples:
        >>> info("Loading libstempo")
    """
    configure_logging()
    _LOGGER.info(msg)
