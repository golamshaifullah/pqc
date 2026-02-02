"""Detect exponential dip events in timing residuals.

A dip is modeled as a negative amplitude exponential recovery:

    y(t) ≈ A * exp(-(t - t0) / tau_rec) for t >= t0, with A < 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

def scan_exp_dips(
    df: pd.DataFrame,
    *,
    mjd_col: str = "mjd",
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    freq_col: str | None = None,
    exclude_bad_col: str = "bad",
    tau_rec_days: float = 30.0,
    window_mult: float = 5.0,
    min_points: int = 6,
    delta_chi2_thresh: float = 25.0,
    suppress_overlap: bool = True,
    member_eta: float = 1.0,
    freq_dependence: bool = True,
    freq_alpha_min: float = 0.0,
    freq_alpha_max: float = 4.0,
    freq_alpha_tol: float = 1e-3,
    freq_alpha_max_iter: int = 64,
    force_minimum_member: bool = True,
) -> pd.DataFrame:
    """Scan for exponential dip recoveries and annotate affected rows.

    Adds columns: exp_dip_id, exp_dip_amp, exp_dip_t0, exp_dip_delta_chi2,
    exp_dip_member.
    """
    d = df.sort_values(mjd_col).copy()
    d["exp_dip_id"] = -1
    d["exp_dip_amp"] = np.nan
    d["exp_dip_t0"] = np.nan
    d["exp_dip_delta_chi2"] = np.nan
    d["exp_dip_member"] = False
    d["exp_dip_alpha"] = np.nan

    use = np.ones(len(d), dtype=bool)
    if exclude_bad_col in d.columns:
        use &= ~d[exclude_bad_col].fillna(False).to_numpy()

    t = d[mjd_col].to_numpy(dtype=float)
    y = d[resid_col].to_numpy(dtype=float)
    s = d[sigma_col].to_numpy(dtype=float)
    freq = None
    if freq_dependence:
        if freq_col is None or freq_col not in d.columns:
            return d
        freq = pd.to_numeric(d[freq_col], errors="coerce").to_numpy(dtype=float)

    cand = np.where(use)[0]
    if len(cand) < min_points:
        return d

    w_end = window_mult * tau_rec_days
    def _delta_for_alpha(alpha: float, tt: np.ndarray, yy: np.ndarray, ww: np.ndarray, ff: np.ndarray) -> float:
        f = np.exp(-tt / tau_rec_days) / (ff ** alpha)
        denom = np.sum(ww * f * f)
        if denom <= 0:
            return -np.inf
        A = np.sum(ww * f * yy) / denom
        if not np.isfinite(A) or A >= 0:
            return -np.inf
        chi2_null = np.sum(ww * (yy ** 2))
        chi2_model = np.sum(ww * ((yy - A * f) ** 2))
        return chi2_null - chi2_model

    def _optimize_alpha(
        tt: np.ndarray, yy: np.ndarray, ww: np.ndarray, ff: np.ndarray
    ) -> tuple[float, float]:
        a = float(freq_alpha_min)
        b = float(freq_alpha_max)
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            return np.nan, -np.inf
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _delta_for_alpha(c, tt, yy, ww, ff)
        fd = _delta_for_alpha(d, tt, yy, ww, ff)
        for _ in range(int(freq_alpha_max_iter)):
            if abs(b - a) <= float(freq_alpha_tol):
                break
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = _delta_for_alpha(c, tt, yy, ww, ff)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = _delta_for_alpha(d, tt, yy, ww, ff)
        if fc >= fd:
            return c, fc
        return d, fd

    events: list[tuple[float, float, float, np.ndarray, float]] = []

    for idx0 in cand:
        t0 = t[idx0]
        in_win = use & (t >= t0) & (t <= t0 + w_end)
        if np.count_nonzero(in_win) < min_points:
            continue

        tt = t[in_win] - t0
        yy = y[in_win]
        ww = 1.0 / (s[in_win] ** 2)

        best = None
        if freq_dependence and freq is not None:
            ff = freq[in_win]
            goodf = np.isfinite(ff) & (ff != 0)
            if np.count_nonzero(goodf) < min_points:
                continue
            tt = tt[goodf]
            yy = yy[goodf]
            ww = ww[goodf]
            ff = ff[goodf]
            alpha, delta = _optimize_alpha(tt, yy, ww, ff)
            if np.isfinite(alpha) and np.isfinite(delta):
                f = np.exp(-tt / tau_rec_days) / (ff ** alpha)
                denom = np.sum(ww * f * f)
                if denom > 0:
                    A = np.sum(ww * f * yy) / denom
                    if np.isfinite(A) and A < 0:
                        best = (A, delta, alpha)
        else:
            f = np.exp(-tt / tau_rec_days)
            denom = np.sum(ww * f * f)
            if denom > 0:
                A = np.sum(ww * f * yy) / denom
                if np.isfinite(A) and A < 0:
                    chi2_null = np.sum(ww * (yy ** 2))
                    chi2_model = np.sum(ww * ((yy - A * f) ** 2))
                    delta = chi2_null - chi2_model
                    best = (A, delta, np.nan)

        if best is None:
            continue
        A, delta, alpha = best
        if delta >= delta_chi2_thresh:
            events.append((t0, A, delta, in_win.copy(), float(alpha)))

    if not events:
        return d

    events.sort(key=lambda e: e[2], reverse=True)

    assigned = np.zeros(len(d), dtype=bool)
    kept = []

    for t0, A, delta, in_win, alpha in events:
        if suppress_overlap and np.any(assigned & in_win):
            continue
        kept.append((t0, A, delta, in_win, alpha))
        assigned |= in_win

    for k, (t0, A, delta, in_win, alpha) in enumerate(kept):
        tt = t[in_win] - t0
        if freq_dependence and freq is not None:
            ff = freq[in_win]
            goodf = np.isfinite(ff) & (ff != 0)
            f = np.full_like(tt, np.nan, dtype=float)
            f[goodf] = np.exp(-tt[goodf] / tau_rec_days) / (ff[goodf] ** alpha)
        else:
            f = np.exp(-tt / tau_rec_days)
        model = A * f
        z_pt = np.full_like(t, np.nan, dtype=float)
        sig = s[in_win]
        good = np.isfinite(sig) & (sig > 0)
        if np.any(good):
            z_pt[in_win] = np.where(good, np.abs(model) / sig, np.nan)
        member = in_win.copy()
        if np.isfinite(member_eta):
            member &= (z_pt >= float(member_eta))
        # Ensure the seed (t0) is always included when it is a valid point.
        if force_minimum_member:
            member[idx0] = True

        d.loc[member, "exp_dip_id"] = k
        d.loc[member, "exp_dip_amp"] = A
        d.loc[member, "exp_dip_t0"] = t0
        d.loc[member, "exp_dip_delta_chi2"] = delta
        d.loc[member, "exp_dip_member"] = True
        d.loc[member, "exp_dip_alpha"] = alpha

    return d
