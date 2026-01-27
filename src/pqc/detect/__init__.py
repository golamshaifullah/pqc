"""Provide detection algorithms for PTA QC.

This subpackage groups statistical detection routines used by
:func:`pqc.pipeline.run_pipeline`.

Modules include:
    - OU innovations and noise estimation (:mod:`pqc.detect.ou`).
    - Bad measurement detection (:mod:`pqc.detect.bad_measurements`).
    - Transient exponential recovery scans (:mod:`pqc.detect.transients`).
    - Feature-domain structure tests (:mod:`pqc.detect.feature_structure`).
"""
