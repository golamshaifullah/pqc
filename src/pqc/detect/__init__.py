"""Provide detection algorithms for PTA QC.

This subpackage groups statistical detection routines used by
:func:`pqc.pipeline.run_pipeline`, including OU-based outlier detection,
transient scans, exponential dip scans, and feature-domain structure diagnostics.

See Also:
    pqc.pipeline.run_pipeline: Pipeline entry point that calls detectors.
    pqc.detect.ou: OU innovations and noise estimation.
    pqc.detect.bad_measurements: Bad measurement detection.
    pqc.detect.transients: Transient exponential recovery scans.
    pqc.detect.exp_dips: Exponential dip recovery scans.
    pqc.detect.solar_events: Solar elongation event scans.
    pqc.detect.eclipse_events: Orbital-phase eclipse event scans.
    pqc.detect.gaussian_bumps: Global Gaussian-bump event scans.
    pqc.detect.glitches: Global glitch event scans.
    pqc.detect.feature_structure: Feature-domain structure tests.
"""
