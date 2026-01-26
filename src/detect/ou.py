"""OU innovations and noise-scale estimation for irregular sampling.

We model short-term correlation via an Ornstein–Uhlenbeck (OU) process with
correlation timescale ``tau_days`` and white-noise variance ``q``.
"""

from __future__ import annotations
import numpy as np
from pqc.utils.stats import robust_scale_mad

def ou_innovations_z(t_days, y, sigma, tau_days, q):
    """Compute normalized OU innovations for irregularly sampled data.

    Args:
        t_days: Observation times (days).
        y: Residual values aligned with ``t_days``.
        sigma: Measurement uncertainties for ``y``.
        tau_days: OU correlation timescale (days).
        q: Additional white-noise variance term.

    Returns:
        Array of normalized innovations ``z`` with ``NaN`` for invalid entries.
    """
    t = np.asarray(t_days, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.asarray(sigma, dtype=float)

    n = len(t)
    z = np.full(n, np.nan, dtype=float)
    if n == 0:
        return z

    v0 = s[0] ** 2 + q
    z[0] = y[0] / np.sqrt(v0) if v0 > 0 else np.nan

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        phi = np.exp(-dt / tau_days) if tau_days > 0 else 0.0
        innov = y[i] - phi * y[i - 1]
        vinnov = (s[i] ** 2) + (phi ** 2) * (s[i - 1] ** 2) + q * (1.0 - phi ** 2)
        z[i] = innov / np.sqrt(vinnov) if vinnov > 0 else np.nan
    return z

def estimate_q(t_days, y, sigma, tau_days, q_max_factor=100.0):
    """Estimate ``q`` by matching the robust scale of innovations to unity.

    A binary search selects ``q >= 0`` so the MAD-based scale of ``z`` is
    approximately 1.0.

    Args:
        t_days: Observation times (days).
        y: Residual values aligned with ``t_days``.
        sigma: Measurement uncertainties for ``y``.
        tau_days: OU correlation timescale (days).
        q_max_factor: Upper bound multiplier relative to median ``sigma^2``.

    Returns:
        Estimated non-negative ``q`` value.
    """
    t = np.asarray(t_days, dtype=float)
    y = np.asarray(y, dtype=float)
    s = np.asarray(sigma, dtype=float)

    def scale_minus_one(q):
        z = ou_innovations_z(t, y, s, tau_days, q)
        z = z[np.isfinite(z)]
        if len(z) < 10:
            return 0.0
        return robust_scale_mad(z) - 1.0

    q_lo = 0.0
    q_hi = np.nanmedian(s ** 2) * q_max_factor

    f_lo = scale_minus_one(q_lo)
    if f_lo <= 0:
        return 0.0

    f_hi = scale_minus_one(q_hi)
    if f_hi > 0:
        return q_hi

    for _ in range(50):
        q_mid = 0.5 * (q_lo + q_hi)
        f_mid = scale_minus_one(q_mid)
        if f_mid > 0:
            q_lo = q_mid
        else:
            q_hi = q_mid
    return q_hi
