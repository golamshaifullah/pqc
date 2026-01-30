"""Compute OU innovations and noise-scale estimates for irregular sampling.

We model short-term correlation via an Ornstein–Uhlenbeck (OU) process with
correlation timescale ``tau_days`` and white-noise variance ``q``.

See Also:
    pqc.detect.bad_measurements.detect_bad: Uses OU innovations for outlier tests.
    pqc.utils.stats.robust_scale_mad: Robust scale estimator used for q fitting.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pqc.utils.stats import robust_scale_mad


def ou_innovations_z(
    t_days: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    tau_days: float,
    q: float,
) -> np.ndarray:
    """Compute normalized OU innovations for irregularly sampled data.

    Args:
        t_days (np.ndarray | Sequence[float]): Observation times (days).
        y (np.ndarray | Sequence[float]): Residual values aligned with
            ``t_days``.
        sigma (np.ndarray | Sequence[float]): Measurement uncertainties for
            ``y``.
        tau_days (float): OU correlation timescale (days).
        q (float): Additional white-noise variance term.

    Returns:
        np.ndarray: Normalized innovations ``z`` with ``NaN`` for invalid
        entries.

    Notes:
        The returned array is aligned with the input ordering of ``t_days`` and
        assumes adjacent samples are time-ordered.

    Examples:
        >>> ou_innovations_z([0.0, 1.0], [0.1, 0.0], [1.0, 1.0], tau_days=10.0, q=0.0).shape
        (2,)
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
        vinnov = (s[i] ** 2) + (phi**2) * (s[i - 1] ** 2) + q * (1.0 - phi**2)
        z[i] = innov / np.sqrt(vinnov) if vinnov > 0 else np.nan
    return z


def estimate_q(
    t_days: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    tau_days: float,
    q_max_factor: float = 100.0,
) -> float:
    """Estimate ``q`` by matching the robust scale of innovations to unity.

    A binary search selects ``q >= 0`` so the MAD-based scale of ``z`` is
    approximately 1.0.

    Args:
        t_days (np.ndarray | Sequence[float]): Observation times (days).
        y (np.ndarray | Sequence[float]): Residual values aligned with
            ``t_days``.
        sigma (np.ndarray | Sequence[float]): Measurement uncertainties for
            ``y``.
        tau_days (float): OU correlation timescale (days).
        q_max_factor (float): Upper bound multiplier relative to median
            ``sigma^2``.

    Returns:
        float: Estimated non-negative ``q`` value.

    Notes:
        If the innovations already have robust scale ≤ 1, the estimate is 0.

    Examples:
        >>> estimate_q([0.0, 1.0, 2.0], [0.1, 0.0, -0.1], [1.0, 1.0, 1.0], tau_days=10.0) >= 0.0
        True
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
    q_hi = np.nanmedian(s**2) * q_max_factor

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
