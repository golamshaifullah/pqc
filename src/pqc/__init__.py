"""Provide PTA residual quality-control utilities.

PQC implements a lightweight workflow for quality control (QC) of pulsar
timing array (PTA) residuals. The package is intentionally small and focuses
on a single end-to-end pipeline that loads timing data, merges metadata, and
annotates TOAs with QC flags and diagnostics.

Key capabilities include:
    - Parsing tempo2 timfiles (including INCLUDE recursion and flags).
    - Loading TOA/residual arrays via libstempo.
    - Merging timing arrays with timfile metadata.
    - Normalizing backend keys (``sys``/``group``) for per-backend analysis.
    - Detecting bad measurements and transient exponential recoveries.
    - Optional feature extraction and feature-domain structure diagnostics.

Most users should start with :func:`pqc.pipeline.run_pipeline` or the CLI
entry point in :mod:`pqc.cli`.

See Also:
    pqc.pipeline.run_pipeline: End-to-end QC pipeline for a single pulsar.
    pqc.cli.main: CLI entry point for batch or scripted runs.
"""

__all__ = []
