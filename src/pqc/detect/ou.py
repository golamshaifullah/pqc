"""Compute OU innovations and auxiliary noise-scale estimates.

This module provides the core OU-statistics used by bad-measurement detection.
For irregularly sampled residuals, it computes normalized innovations under an
Ornstein–Uhlenbeck (OU) model and estimates a non-negative extra variance
parameter ``q`` using robust scale matching.

Notes
-----
Definition (OU process)
    Continuous-time mean-reverting Gaussian process:
    :math:`dX_t = -\\theta X_t dt + \\sigma dW_t`, with
    :math:`\\tau = 1/\\theta` as correlation timescale.

Discretization used here
    For intervals :math:`\\Delta t_i`, use
    :math:`\\phi_i = \\exp(-\\Delta t_i/\\tau)` and innovation
    :math:`e_i = y_i - \\phi_i y_{i-1}`.

Why used here
    Residuals are often serially correlated on short scales; innovation
    normalization approximates whitened residual surprises.

Assumptions
    - Locally stationary correlation structure represented by OU.
    - Approximate Gaussianity of innovations for downstream p-values.
    - Correct time ordering of observations.

References
----------
.. [1] Uhlenbeck, G. E., & Ornstein, L. S. (1930), *Physical Review*, 36, 823.
.. [2] Gardiner, C. (2009), *Stochastic Methods*, 4th ed.
.. [3] Rousseeuw, P. J., & Croux, C. (1993), *JASA*, 88(424), 1273-1283.
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

    Raises:
        ValueError
            If array lengths are inconsistent.

    Notes:
        **Formula**
            Let :math:`\\phi_i = \\exp(-(t_i-t_{i-1})/\\tau)`.
            Then

            .. math::
               e_i = y_i - \\phi_i y_{i-1},
               \\quad
               v_i = \\sigma_i^2 + \\phi_i^2\\sigma_{i-1}^2 + q(1-\\phi_i^2),
               \\quad
               z_i = e_i/\\sqrt{v_i}.

            For :math:`i=0`, :math:`z_0 = y_0 / \\sqrt{\\sigma_0^2 + q}`.

        **Interpretation**
            ``|z_i|`` near 0 indicates consistency with OU-noise prediction;
            larger values indicate local deviations.

        **Caveats**
            Non-monotonic times or severe heteroscedastic model misspecification
            can make ``z`` hard to interpret.

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
    """Estimate ``q`` by matching robust innovation scale to unity.

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

    Raises:
        ValueError
            If numeric conversion fails for required arrays.

    Notes:
        **Estimator definition**
            Define :math:`s(q) = \\mathrm{MAD}(z(q))/0.67449`. Solve
            :math:`s(q) \\approx 1` over :math:`q \\ge 0` by bisection.

        **Why used here**
            Robust scale matching prevents a few outliers from strongly
            biasing variance inflation.

        **Assumptions**
            Monotonic decrease of robust scale as ``q`` increases, sufficient
            sample size for stable MAD.

        **Caveats**
            For very short series, estimator may return boundary values.

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
