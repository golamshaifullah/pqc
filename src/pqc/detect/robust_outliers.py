"""Robust outlier detection using median/MAD standardized residuals.

This detector computes robust z-scores using the median and median absolute
deviation (MAD), then flags points exceeding a configurable threshold.

Notes
-----
Definition
    :math:`\\mathrm{MAD}=\\mathrm{median}(|y_i-\\mathrm{median}(y)|)`.
    Robust z-score is approximated as
    :math:`z_i = 0.6745 (y_i-\\mathrm{median}(y))/\\mathrm{MAD}`.

Why used here
    Median/MAD is resilient to contamination and provides a lightweight
    fallback detector when distribution tails are uncertain.

Assumptions
    - Majority of points are not gross outliers.
    - Central tendency is reasonably represented by median.

Caveats
    MAD can be zero for low-variance or quantized series, in which case no
    robust outlier is reported.

References
----------
.. [1] Hampel, F. R. (1974), *JASA* 69(346), 383-393.
.. [2] Rousseeuw, P. J., & Croux, C. (1993), *JASA* 88(424), 1273-1283.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_robust_outliers(
    df: pd.DataFrame,
    *,
    resid_col: str = "resid",
    sigma_col: str | None = None,
    z_thresh: float = 5.0,
    prefix: str = "robust",
) -> pd.DataFrame:
    """Flag robust outliers using median/MAD standardized scores.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    resid_col : str, optional
        Residual column to analyze.
    sigma_col : str or None, optional
        Reserved for interface compatibility; currently unused.
    z_thresh : float, optional
        Absolute robust-z threshold for labeling outliers.
    prefix : str, optional
        Prefix for output columns.

    Returns
    -------
    pandas.DataFrame
        Copy with added columns ``<prefix>_z`` and ``<prefix>_outlier``.

    Notes
    -----
    Worked example
        If ``median=0``, ``MAD=2e-7``, and a point has residual ``2e-6``, then
        :math:`z\\approx 0.6745\\times 10 = 6.745`; with ``z_thresh=5`` this
        point is flagged.
    """
    out = df.copy()
    out[f"{prefix}_z"] = np.nan
    out[f"{prefix}_outlier"] = False

    if resid_col not in out.columns:
        return out

    y = pd.to_numeric(out[resid_col], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(y)
    if not np.any(good):
        return out

    med = np.nanmedian(y[good])
    mad = np.nanmedian(np.abs(y[good] - med))
    if not np.isfinite(mad) or mad == 0:
        return out

    z = 0.6745 * (y - med) / mad
    out.loc[good, f"{prefix}_z"] = z[good]
    out.loc[good, f"{prefix}_outlier"] = np.abs(z[good]) >= float(z_thresh)
    return out
