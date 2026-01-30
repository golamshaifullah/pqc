"""Provide feature engineering utilities for PTA QC.

This subpackage adds per-TOA covariates (e.g., orbital phase, solar
elongation, observatory geometry) that can be used for feature-domain
diagnostics and detrending.

See Also:
    pqc.pipeline.run_pipeline: Pipeline entry point that can attach features.
    pqc.features.feature_extraction: Feature extraction helpers.
    pqc.features.backend_keys: Backend key normalization utilities.
"""

from pqc.features.feature_extraction import (
    add_altaz_features,
    add_feature_columns,
    add_freq_bin,
    add_orbital_phase,
    add_solar_elongation,
)

__all__ = [
    "add_feature_columns",
    "add_altaz_features",
    "add_freq_bin",
    "add_orbital_phase",
    "add_solar_elongation",
]
