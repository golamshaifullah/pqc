"""Detect achromatic and DM-like step changes in timing residuals.

This module fits single changepoint step models using weighted two-segment
means and a likelihood-ratio-style statistic expressed as :math:`\\Delta\\chi^2`.
It supports:

- achromatic steps (constant amplitude after ``t0``), and
- DM-like chromatic steps scaling as :math:`1/f^2`.

Notes
-----
Statistic
    For a split at index :math:`i`, let weighted means before/after be
    :math:`\\mu_1, \\mu_2` with variances :math:`1/W_1, 1/W_2`. The step
    contrast is :math:`\\delta=\\mu_2-\\mu_1` and score is
    :math:`\\Delta\\chi^2 = \\delta^2/(1/W_1+1/W_2)`.

Why used here
    Step discontinuities are common signatures of timing-model offsets or
    abrupt propagation/instrument changes.

Assumptions
    - One dominant changepoint in the tested series.
    - Gaussian errors with known/estimated ``sigma``.
    - Sorted times and enough points on each side of split.

References
----------
.. [1] Lorimer, D. R., & Kramer, M. (2005), *Handbook of Pulsar Astronomy*.
.. [2] Edwards, R. T., Hobbs, G. B., & Manchester, R. N. (2006), *MNRAS* 372.
.. [3] Killick, R., Fearnhead, P., & Eckley, I. A. (2012), *JASA* 107(500),
   1590-1598.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _best_step(
    mjd: np.ndarray, y: np.ndarray, s: np.ndarray, min_points: int, delta_chi2_thresh: float
):
    n = len(y)
    if n < 2 * min_points:
        return None

    w = 1.0 / (s**2)
    # cumulative sums for fast split stats
    w_c = np.cumsum(w)
    wy_c = np.cumsum(w * y)

    best = None
    for i in range(min_points, n - min_points + 1):
        w1 = w_c[i - 1]
        wy1 = wy_c[i - 1]
        w2 = w_c[-1] - w1
        wy2 = wy_c[-1] - wy1
        if w1 <= 0 or w2 <= 0:
            continue
        mu1 = wy1 / w1
        mu2 = wy2 / w2
        var1 = 1.0 / w1
        var2 = 1.0 / w2
        delta = mu2 - mu1
        chi2 = (delta * delta) / (var1 + var2)
        if best is None or chi2 > best["delta_chi2"]:
            t0 = 0.5 * (mjd[i - 1] + mjd[i])
            best = {"idx": i, "t0": t0, "amp": delta, "delta_chi2": chi2}

    if best is None or best["delta_chi2"] < delta_chi2_thresh:
        return None
    return best


def detect_step(
    df: pd.DataFrame,
    *,
    mjd_col: str = "mjd",
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    min_points: int = 20,
    delta_chi2_thresh: float = 25.0,
    member_eta: float = 1.0,
    member_tmax_days: float | None = 3650.0,
    instrument: bool = False,
    prefix: str = "step",
) -> pd.DataFrame:
    """Detect a single achromatic step-like offset and annotate rows.

    Returns a DataFrame with columns: <prefix>_id, <prefix>_t0, <prefix>_amp,
    <prefix>_delta_chi2. Rows are marked with id=0 for mjd >= t0.

    Parameters
    ----------
    df : pandas.DataFrame
        Input timing table.
    mjd_col, resid_col, sigma_col : str, optional
        Column names for time, residual, and uncertainty.
    min_points : int, optional
        Minimum points required on each side of candidate split.
    delta_chi2_thresh : float, optional
        Minimum :math:`\\Delta\\chi^2` to accept a step candidate.
    member_eta : float, optional
        Membership threshold on :math:`|m_i|/\\sigma_i`.
    member_tmax_days : float or None, optional
        Maximum post-step membership horizon in days.
    instrument : bool, optional
        Emit diagnostics if True.
    prefix : str, optional
        Prefix for output columns.

    Returns
    -------
    pandas.DataFrame
        Copy of input with step annotations and membership diagnostics.

    Notes
    -----
    Worked example
        With ``amp=2e-6`` s and ``sigma=5e-7`` s, per-point membership score
        is :math:`z=|amp|/sigma=4`. For ``member_eta=1``, such points are
        informative members.
    """
    out = df.copy()
    out[f"{prefix}_id"] = -1
    out[f"{prefix}_applicable"] = False
    out[f"{prefix}_informative"] = False
    out[f"{prefix}_t0"] = np.nan
    out[f"{prefix}_amp"] = np.nan
    out[f"{prefix}_delta_chi2"] = np.nan
    out[f"{prefix}_n_applicable"] = 0
    out[f"{prefix}_n_informative"] = 0
    out[f"{prefix}_frac_informative_z_lt1"] = 0.0

    cols = [mjd_col, resid_col, sigma_col]
    if any(c not in out.columns for c in cols):
        return out

    d = out[cols].dropna().copy()
    if d.empty:
        return out

    d = d.sort_values(mjd_col)
    mjd = d[mjd_col].to_numpy(dtype=float)
    y = d[resid_col].to_numpy(dtype=float)
    s = d[sigma_col].to_numpy(dtype=float)

    best = _best_step(mjd, y, s, int(min_points), float(delta_chi2_thresh))
    if best is None:
        return out

    t0 = float(best["t0"])
    amp = float(best["amp"])
    t = out[mjd_col].to_numpy(dtype=float)
    s = out[sigma_col].to_numpy(dtype=float)
    t_end = np.nanmax(t) if member_tmax_days is None else (t0 + float(member_tmax_days))
    applicable = np.isfinite(t) & (t >= t0) & (t <= t_end)
    effect = np.zeros_like(t, dtype=float)
    effect[applicable] = np.abs(amp)
    z_pt = np.full_like(t, np.nan, dtype=float)
    good = applicable & np.isfinite(s) & (s > 0)
    z_pt[good] = effect[good] / s[good]
    informative = applicable.copy()
    if np.isfinite(member_eta):
        informative &= z_pt > float(member_eta)

    out.loc[applicable, f"{prefix}_id"] = 0
    out[f"{prefix}_applicable"] = applicable
    out[f"{prefix}_informative"] = informative
    out[f"{prefix}_t0"] = t0
    out[f"{prefix}_amp"] = amp
    out[f"{prefix}_delta_chi2"] = float(best["delta_chi2"])

    z_inf = z_pt[informative]
    out[f"{prefix}_n_applicable"] = int(np.count_nonzero(applicable))
    out[f"{prefix}_n_informative"] = int(np.count_nonzero(informative))
    out[f"{prefix}_n_members"] = int(np.count_nonzero(informative))
    if len(z_inf):
        out[f"{prefix}_frac_informative_z_lt1"] = float(np.mean(z_inf < 1.0))
        out[f"{prefix}_frac_z_lt1"] = float(np.mean(z_inf < 1.0))
        out[f"{prefix}_z_min"] = float(np.nanmin(z_inf))
        out[f"{prefix}_z_med"] = float(np.nanmedian(z_inf))
        out[f"{prefix}_z_max"] = float(np.nanmax(z_inf))
    else:
        out[f"{prefix}_frac_informative_z_lt1"] = 0.0
        out[f"{prefix}_frac_z_lt1"] = 0.0
        out[f"{prefix}_z_min"] = np.nan
        out[f"{prefix}_z_med"] = np.nan
        out[f"{prefix}_z_max"] = np.nan

    if instrument:
        zf = z_pt[np.isfinite(z_pt)]
        if len(zf):
            info_str = (
                f"{prefix}_id=0 t0={t0:.6f} amp={amp:.3g} "
                f"n_applicable={int(np.count_nonzero(applicable))} n_informative={int(np.count_nonzero(informative))} "
                f"z_pt[min/med/max]={np.nanmin(zf):.3g}/{np.nanmedian(zf):.3g}/{np.nanmax(zf):.3g} "
                f"frac<1={float(np.mean(zf < 1.0)):.3g} frac<2={float(np.mean(zf < 2.0)):.3g}"
            )
            try:
                from pqc.utils.logging import info

                info(info_str)
                if np.mean(zf < 1.0) > 0.5:
                    from pqc.utils.logging import warn

                    warn(
                        f"{prefix} membership has >50% members with z_pt<1.0; check membership criteria."
                    )
            except Exception as exc:
                from pqc.utils.logging import warn

                warn(f"{prefix} instrumentation logging failed: {exc}")
    return out


def detect_dm_step(
    df: pd.DataFrame,
    *,
    mjd_col: str = "mjd",
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    freq_col: str = "freq",
    min_points: int = 20,
    delta_chi2_thresh: float = 25.0,
    member_eta: float = 1.0,
    member_tmax_days: float | None = 3650.0,
    instrument: bool = False,
    prefix: str = "dm_step",
) -> pd.DataFrame:
    """Detect chromatic step offsets consistent with DM-like scaling.

    Uses residuals scaled by 1/freq^2 to normalize dispersion.

    Parameters
    ----------
    df : pandas.DataFrame
        Input timing table.
    mjd_col, resid_col, sigma_col, freq_col : str, optional
        Column names for required inputs.
    min_points : int, optional
        Minimum points required on each side of split.
    delta_chi2_thresh : float, optional
        Acceptance threshold in :math:`\\Delta\\chi^2`.
    member_eta : float, optional
        Membership threshold on :math:`|m_i|/\\sigma_i`.
    member_tmax_days : float or None, optional
        Maximum post-step membership horizon in days.
    instrument : bool, optional
        Emit diagnostics if True.
    prefix : str, optional
        Prefix for output columns.

    Returns
    -------
    pandas.DataFrame
        Copy of input with DM-step annotations and membership diagnostics.

    Notes
    -----
    Formula
        For frequency :math:`f`, model effect after ``t0`` is
        :math:`m_i = A/f_i^2`. Membership score is
        :math:`z_i = |A|/(f_i^2\\sigma_i)`.

    Caveats
    -------
    Incorrect frequency metadata or non-dispersive chromatic effects can
    mimic/obscure DM-step detection.
    """
    out = df.copy()
    out[f"{prefix}_id"] = -1
    out[f"{prefix}_applicable"] = False
    out[f"{prefix}_informative"] = False
    out[f"{prefix}_t0"] = np.nan
    out[f"{prefix}_amp"] = np.nan
    out[f"{prefix}_delta_chi2"] = np.nan
    out[f"{prefix}_n_applicable"] = 0
    out[f"{prefix}_n_informative"] = 0
    out[f"{prefix}_frac_informative_z_lt1"] = 0.0

    cols = [mjd_col, resid_col, sigma_col, freq_col]
    if any(c not in out.columns for c in cols):
        return out

    d = out[cols].dropna().copy()
    if d.empty:
        return out

    freq = pd.to_numeric(d[freq_col], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(freq) & (freq != 0)
    if not np.any(good):
        return out

    d = d.loc[good]
    invf2 = 1.0 / (freq[good] ** 2)
    y = d[resid_col].to_numpy(dtype=float) / invf2
    s = d[sigma_col].to_numpy(dtype=float) / invf2
    mjd = d[mjd_col].to_numpy(dtype=float)

    # Run step detection on DM-scaled residuals
    best = _best_step(mjd, y, s, int(min_points), float(delta_chi2_thresh))
    if best is None:
        return out

    t0 = float(best["t0"])
    amp = float(best["amp"])
    t_all = out[mjd_col].to_numpy(dtype=float)
    s_all = out[sigma_col].to_numpy(dtype=float)
    freq_all = pd.to_numeric(out[freq_col], errors="coerce").to_numpy(dtype=float)
    t_end = np.nanmax(t_all) if member_tmax_days is None else (t0 + float(member_tmax_days))
    applicable = np.isfinite(t_all) & (t_all >= t0) & (t_all <= t_end)
    z_pt = np.full_like(t_all, np.nan, dtype=float)
    model = np.full_like(t_all, np.nan, dtype=float)
    good = applicable & np.isfinite(s_all) & (s_all > 0) & np.isfinite(freq_all) & (freq_all != 0)
    model[good] = amp / (freq_all[good] ** 2)
    z_pt[good] = np.abs(model[good]) / s_all[good]
    informative = applicable.copy()
    if np.isfinite(member_eta):
        informative &= z_pt > float(member_eta)

    out.loc[applicable, f"{prefix}_id"] = 0
    out[f"{prefix}_applicable"] = applicable
    out[f"{prefix}_informative"] = informative
    out[f"{prefix}_t0"] = t0
    out[f"{prefix}_amp"] = amp
    out[f"{prefix}_delta_chi2"] = float(best["delta_chi2"])

    z_inf = z_pt[informative]
    out[f"{prefix}_n_applicable"] = int(np.count_nonzero(applicable))
    out[f"{prefix}_n_informative"] = int(np.count_nonzero(informative))
    out[f"{prefix}_n_members"] = int(np.count_nonzero(informative))
    if len(z_inf):
        out[f"{prefix}_frac_informative_z_lt1"] = float(np.mean(z_inf < 1.0))
        out[f"{prefix}_frac_z_lt1"] = float(np.mean(z_inf < 1.0))
        out[f"{prefix}_z_min"] = float(np.nanmin(z_inf))
        out[f"{prefix}_z_med"] = float(np.nanmedian(z_inf))
        out[f"{prefix}_z_max"] = float(np.nanmax(z_inf))
    else:
        out[f"{prefix}_frac_informative_z_lt1"] = 0.0
        out[f"{prefix}_frac_z_lt1"] = 0.0
        out[f"{prefix}_z_min"] = np.nan
        out[f"{prefix}_z_med"] = np.nan
        out[f"{prefix}_z_max"] = np.nan

    if instrument:
        zf = z_pt[np.isfinite(z_pt)]
        if len(zf):
            info_str = (
                f"{prefix}_id=0 t0={t0:.6f} amp={amp:.3g} "
                f"n_applicable={int(np.count_nonzero(applicable))} n_informative={int(np.count_nonzero(informative))} "
                f"z_pt[min/med/max]={np.nanmin(zf):.3g}/{np.nanmedian(zf):.3g}/{np.nanmax(zf):.3g} "
                f"frac<1={float(np.mean(zf < 1.0)):.3g} frac<2={float(np.mean(zf < 2.0)):.3g}"
            )
            try:
                from pqc.utils.logging import info

                info(info_str)
                if np.mean(zf < 1.0) > 0.5:
                    from pqc.utils.logging import warn

                    warn(
                        f"{prefix} membership has >50% members with z_pt<1.0; check membership criteria."
                    )
            except Exception as exc:
                from pqc.utils.logging import warn

                warn(f"{prefix} instrumentation logging failed: {exc}")
    return out
