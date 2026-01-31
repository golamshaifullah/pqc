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
    pqc.detect.feature_structure: Feature-domain structure tests.
"""
