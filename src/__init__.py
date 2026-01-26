"""PTA residual quality-control utilities.

This package implements a lightweight QC workflow for pulsar timing residuals.
It focuses on:

- Parsing tempo2 timfiles (including INCLUDE recursion and flags).
- Loading TOA/residual arrays via libstempo.
- Merging timing arrays with timfile metadata.
- Normalizing backend keys (``sys``/``group``) for per-backend analysis.
- Detecting bad measurements and transient exponential recoveries.

Typical usage is via :func:`pqc.pipeline.run_pipeline` or the CLI entry point
in :mod:`pqc.cli`.
"""

__all__ = []
