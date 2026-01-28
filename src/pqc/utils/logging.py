"""Provide minimal logging helpers for CLI-style output.

These helpers intentionally avoid the standard :mod:`logging` module to keep
CLI output terse and easily redirectable. They are suitable for scripts but do
not provide log levels or configuration.
"""

from __future__ import annotations
import sys

def warn(msg: str) -> None:
    """Print a warning message to stderr.

    Args:
        msg (str): Warning message text.

    Examples:
        >>> warn("Missing metadata")
    """
    print(f"WARNING: {msg}", file=sys.stderr)

def info(msg: str) -> None:
    """Print an informational message to stdout.

    Args:
        msg (str): Message text.

    Examples:
        >>> info("Loading libstempo")
    """
    print(msg, file=sys.stdout)
