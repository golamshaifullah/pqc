"""Provide small statistical utilities without SciPy.

These helpers implement small, reusable statistical transforms used by the
detectors. They intentionally avoid SciPy to keep dependencies minimal.
"""

from __future__ import annotations
import numpy as np
from math import erfc

def norm_abs_sf(x: np.ndarray) -> np.ndarray:
    """Return survival probabilities for |Z| where Z ~ N(0,1).

    Args:
        x: Array of non-negative z-values.

    Returns:
        Array of probabilities ``P(|Z| >= x)``.

    Notes:
        This function uses the complementary error function from the standard
        library to avoid SciPy.

    Examples:
        >>> norm_abs_sf(np.array([0.0, 1.0])).round(6)
        array([1.      , 0.317311])
    """
    x = np.asarray(x, dtype=float)
    return np.vectorize(erfc, otypes=[float])(x / np.sqrt(2.0))

def bh_fdr(pvals: np.ndarray, q: float) -> np.ndarray:
    """Apply Benjamini–Hochberg FDR and return discovery mask.

    Args:
        pvals: Array of p-values.
        q: Target false discovery rate.

    Returns:
        Boolean array indicating discoveries.

    Notes:
        The returned mask is aligned with the input ordering of ``pvals``.

    Examples:
        >>> bh_fdr(np.array([0.001, 0.2, 0.03]), q=0.05)
        array([ True, False,  True])
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(p)
    p_sorted = p[order]
    thresh = q * (np.arange(1, m + 1) / m)

    passed = p_sorted <= thresh
    if not np.any(passed):
        return np.zeros(m, dtype=bool)

    k = np.max(np.where(passed)[0])
    cutoff = p_sorted[k]
    return p <= cutoff

def robust_scale_mad(x: np.ndarray) -> float:
    """Estimate scale using the median absolute deviation (MAD).

    Args:
        x: Input sample array.

    Returns:
        Robust estimate of standard deviation.

    Notes:
        Uses the standard Gaussian consistency factor (1.4826).

    Examples:
        >>> robust_scale_mad(np.array([0.0, 1.0, 2.0, 100.0])) > 0
        True
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else float(np.std(x))

def chi2_sf_approx(chi2: float, dof: int) -> float:
    """Approximate chi-square survival function using Wilson-Hilferty transform."""
    if dof <= 0 or not np.isfinite(chi2):
        return float("nan")
    if chi2 < 0:
        return 1.0
    k = float(dof)
    z = ((chi2 / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / np.sqrt(2.0 / (9.0 * k))
    return 0.5 * erfc(z / np.sqrt(2.0))
