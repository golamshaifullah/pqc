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
class FeatureConfig:
    """Configure feature-column extraction.

    Attributes:
        add_orbital_phase (bool): If True, compute orbital phase from PB/T0.
        add_solar_elongation (bool): If True, compute solar elongation (requires
            astropy).
        add_elevation (bool): If True, compute elevation (deg) from telescope
            location.
        add_airmass (bool): If True, compute airmass from telescope location.
        add_parallactic_angle (bool): If True, compute parallactic angle (deg).
        add_freq_bin (bool): If True, add a coarse linear frequency-bin index.
        freq_bins (int): Number of bins for ``freq_bin`` if enabled.
        observatory_path (str | None): Optional observatory file with geocentric
            XYZ coords.

    Notes:
        Feature extraction is optional. If astropy is not available, the
        sky-position-based features are filled with NaNs and a warning is
        emitted by the feature extraction helpers.

    Examples:
        >>> FeatureConfig(add_solar_elongation=False, add_freq_bin=True)
    """
    add_orbital_phase: bool = True
    add_solar_elongation: bool = True
    add_elevation: bool = False
    add_airmass: bool = False
    add_parallactic_angle: bool = False
    add_freq_bin: bool = False
    freq_bins: int = 8
    observatory_path: str | None = None

@dataclass(frozen=True)
class StructureConfig:
    """Configure feature-domain structure tests and detrending.

    Attributes:
        mode (str): One of ``none``, ``detrend``, ``test``, or ``both``.
        detrend_features (tuple[str, ...]): Feature columns to detrend against.
        structure_features (tuple[str, ...]): Feature columns to test for
            structure.
        nbins (int): Number of bins for binned means/tests.
        min_per_bin (int): Minimum points per bin to evaluate.
        p_thresh (float): Threshold on approximate chi-square tail probability.
        circular_features (tuple[str, ...]): Feature names treated as circular
            in [0, 1).
        structure_group_cols (tuple[str, ...] | None): Columns used to group
            structure tests/detrending.

    Notes:
        Structure detection is a diagnostic that flags feature-coherent
        behavior without marking individual TOAs as bad.

    Examples:
        >>> StructureConfig(mode="both", nbins=16)
    """
    mode: str = "none"
    detrend_features: tuple[str, ...] = ("solar_elongation_deg", "orbital_phase")
    structure_features: tuple[str, ...] = ("solar_elongation_deg", "orbital_phase")
    nbins: int = 12
    min_per_bin: int = 3
    p_thresh: float = 0.01
    circular_features: tuple[str, ...] = ("orbital_phase",)
    structure_group_cols: tuple[str, ...] | None = None

@dataclass(frozen=True)
class BadMeasConfig:
    """Configure bad-measurement detection.

    Attributes:
        tau_corr_days (float): OU correlation timescale in days.
        fdr_q (float): Benjamini–Hochberg false discovery rate for day-level
            tests.
        mark_only_worst_per_day (bool): If True, mark only the worst TOA per bad
            day.

    Examples:
        Increase the correlation timescale and be more permissive:

        >>> BadMeasConfig(tau_corr_days=0.03, fdr_q=0.02)
    """
    tau_corr_days: float = 0.02     # OU correlation timescale (~30 min)
    fdr_q: float = 0.01             # day-level BH FDR
    mark_only_worst_per_day: bool = True

@dataclass(frozen=True)
class TransientConfig:
    """Configure transient exponential-recovery detection.

    Attributes:
        tau_rec_days (float): Recovery timescale for the exponential decay
            (days).
        window_mult (float): Window length multiplier relative to
            ``tau_rec_days``.
        min_points (int): Minimum points required to consider a candidate
            window.
        delta_chi2_thresh (float): Minimum Δχ² to accept a transient candidate.
        suppress_overlap (bool): If True, prevent overlapping transient
            assignments.

    Examples:
        Look for longer recoveries and stronger events:

        >>> TransientConfig(tau_rec_days=14.0, delta_chi2_thresh=40.0)
    """
    tau_rec_days: float = 7.0       # exp recovery time constant
    window_mult: float = 5.0        # scan window = window_mult * tau_rec
    min_points: int = 6
    delta_chi2_thresh: float = 25.0
    suppress_overlap: bool = True


@dataclass(frozen=True)
class StepConfig:
    """Configure step-like offset detection.

    Attributes:
        enabled (bool): Enable step detection.
        min_points (int): Minimum points on each side of a candidate step.
        delta_chi2_thresh (float): Minimum Δχ² to accept a step.
        scope (str): One of ``backend``, ``global``, or ``both``.

    Notes:
        This detector searches for a single best step per group and annotates
        rows occurring after the step time.
    """
    enabled: bool = True
    min_points: int = 20
    delta_chi2_thresh: float = 25.0
    scope: str = "both"

@dataclass(frozen=True)
class RobustOutlierConfig:
    """Configure robust (MAD-based) outlier detection."""
    enabled: bool = True
    z_thresh: float = 5.0
    scope: str = "global"  # backend/global/both

@dataclass(frozen=True)
class MergeConfig:
    """Configure time/metadata matching.

    Attributes:
        tol_days (float): Maximum allowed |ΔMJD| when matching tim metadata to
            TOAs.

    Examples:
        Use a 3-second merge tolerance:

        >>> MergeConfig(tol_days=3.0 / 86400.0)
    """
    tol_days: float = 2.0 / 86400.0 # 2 seconds default

@dataclass(frozen=True)
class PreprocConfig:
    """Configure covariate-conditioned preprocessing for detectors.

    Attributes:
        detrend_features (tuple[str, ...]): Feature columns to detrend against.
        rescale_feature (str | None): Feature column for variance rescaling.
        condition_on (tuple[str, ...]): Grouping columns for mean/variance models.
        use_preproc_for (tuple[str, ...]): Detectors to run on processed residuals.
        nbins (int): Default number of bins for mean/variance models.
        min_per_bin (int): Minimum points per bin.
        circular_features (tuple[str, ...]): Features treated as circular in [0, 1).
    """
    detrend_features: tuple[str, ...] = ()
    rescale_feature: str | None = None
    condition_on: tuple[str, ...] = ("group",)
    use_preproc_for: tuple[str, ...] = ()
    nbins: int = 12
    min_per_bin: int = 5
    circular_features: tuple[str, ...] = ("orbital_phase",)

@dataclass(frozen=True)
class OutlierGateConfig:
    """Configure hard sigma gating for outlier membership."""
    enabled: bool = False
    sigma_thresh: float = 3.0
    resid_col: str | None = None
    sigma_col: str | None = None
