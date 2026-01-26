"""Small statistical utilities without SciPy."""

from __future__ import annotations
import numpy as np
from math import erfc

def norm_abs_sf(x: np.ndarray) -> np.ndarray:
    """Return survival probabilities for |Z| where Z ~ N(0,1).

    Args:
        x: Array of non-negative z-values.

    Returns:
        Array of probabilities ``P(|Z| >= x)``.
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
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad if mad > 0 else float(np.std(x))
