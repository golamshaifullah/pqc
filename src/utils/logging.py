"""Minimal logging helpers for CLI-style output."""

from __future__ import annotations
import sys

def warn(msg: str) -> None:
    """Print a warning message to stderr.

    Args:
        msg: Warning message text.
    """
    print(f"WARNING: {msg}", file=sys.stderr)

def info(msg: str) -> None:
    """Print an informational message to stdout.

    Args:
        msg: Message text.
    """
    print(msg, file=sys.stdout)
