"""Configuration objects for PTA QC.

This module centralizes tuning knobs used by the QC pipeline. Defaults are
conservative and intended to be safe for a wide range of datasets. Prefer
constructing explicit config objects rather than scattering literals across
modules.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BadMeasConfig:
    """Configuration for bad-measurement detection.

    Attributes:
        tau_corr_days: OU correlation timescale in days.
        fdr_q: Benjamini–Hochberg false discovery rate for day-level tests.
        mark_only_worst_per_day: If True, mark only the worst TOA per bad day.
    """
    tau_corr_days: float = 0.02     # OU correlation timescale (~30 min)
    fdr_q: float = 0.01             # day-level BH FDR
    mark_only_worst_per_day: bool = True

@dataclass(frozen=True)
class TransientConfig:
    """Configuration for transient detection.

    Attributes:
        tau_rec_days: Recovery timescale for the exponential decay (days).
        window_mult: Window length multiplier relative to ``tau_rec_days``.
        min_points: Minimum points required to consider a candidate window.
        delta_chi2_thresh: Minimum Δχ² to accept a transient candidate.
        suppress_overlap: If True, prevent overlapping transient assignments.
    """
    tau_rec_days: float = 7.0       # exp recovery time constant
    window_mult: float = 5.0        # scan window = window_mult * tau_rec
    min_points: int = 6
    delta_chi2_thresh: float = 25.0
    suppress_overlap: bool = True

@dataclass(frozen=True)
class MergeConfig:
    """Configuration for time/metadata matching.

    Attributes:
        tol_days: Maximum allowed |ΔMJD| when matching tim metadata to TOAs.
    """
    tol_days: float = 2.0 / 86400.0 # 2 seconds default
