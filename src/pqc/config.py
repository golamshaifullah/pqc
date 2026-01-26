"""Define configuration objects for the PQC pipeline.

This module centralizes tuning knobs used by the QC pipeline. Defaults are
conservative and intended to be safe for a wide range of datasets. Prefer
constructing explicit config objects rather than scattering literals across
modules.

All configuration classes are frozen dataclasses, making them hashable and
safe to share across runs.

See Also:
    pqc.pipeline.run_pipeline: Consumes these config objects.
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BadMeasConfig:
    """Configure bad-measurement detection.

    Args:
        tau_corr_days: OU correlation timescale in days.
        fdr_q: Benjamini–Hochberg false discovery rate for day-level tests.
        mark_only_worst_per_day: If True, mark only the worst TOA per bad day.

    Attributes:
        tau_corr_days: OU correlation timescale in days.
        fdr_q: Benjamini–Hochberg false discovery rate for day-level tests.
        mark_only_worst_per_day: If True, mark only the worst TOA per bad day.

    Examples:
        Increase the correlation timescale and be more permissive::

            BadMeasConfig(tau_corr_days=0.03, fdr_q=0.02)
    """
    tau_corr_days: float = 0.02     # OU correlation timescale (~30 min)
    fdr_q: float = 0.01             # day-level BH FDR
    mark_only_worst_per_day: bool = True

@dataclass(frozen=True)
class TransientConfig:
    """Configure transient exponential-recovery detection.

    Args:
        tau_rec_days: Recovery timescale for the exponential decay (days).
        window_mult: Window length multiplier relative to ``tau_rec_days``.
        min_points: Minimum points required to consider a candidate window.
        delta_chi2_thresh: Minimum Δχ² to accept a transient candidate.
        suppress_overlap: If True, prevent overlapping transient assignments.

    Attributes:
        tau_rec_days: Recovery timescale for the exponential decay (days).
        window_mult: Window length multiplier relative to ``tau_rec_days``.
        min_points: Minimum points required to consider a candidate window.
        delta_chi2_thresh: Minimum Δχ² to accept a transient candidate.
        suppress_overlap: If True, prevent overlapping transient assignments.

    Examples:
        Look for longer recoveries and stronger events::

            TransientConfig(tau_rec_days=14.0, delta_chi2_thresh=40.0)
    """
    tau_rec_days: float = 7.0       # exp recovery time constant
    window_mult: float = 5.0        # scan window = window_mult * tau_rec
    min_points: int = 6
    delta_chi2_thresh: float = 25.0
    suppress_overlap: bool = True

@dataclass(frozen=True)
class MergeConfig:
    """Configure time/metadata matching.

    Args:
        tol_days: Maximum allowed |ΔMJD| when matching tim metadata to TOAs.

    Attributes:
        tol_days: Maximum allowed |ΔMJD| when matching tim metadata to TOAs.

    Examples:
        Use a 3-second merge tolerance::

            MergeConfig(tol_days=3.0 / 86400.0)
    """
    tol_days: float = 2.0 / 86400.0 # 2 seconds default
