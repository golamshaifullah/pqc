"""Detect solar elongation events in timing residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _year_from_mjd(mjd: np.ndarray) -> np.ndarray:
    # Approximate conversion: year 2000 at MJD 51544.5
    return (2000.0 + (mjd - 51544.5) / 365.25).astype(int)


def detect_solar_events(
    df: pd.DataFrame,
    *,
    mjd_col: str = "mjd",
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    elong_col: str = "solar_elongation_deg",
    freq_col: str | None = "freq",
    enabled: bool = True,
    approach_max_deg: float = 30.0,
    min_points_global: int = 30,
    min_points_year: int = 10,
    min_points_near_zero: int = 3,
    tau_min_deg: float = 2.0,
    tau_max_deg: float = 60.0,
    member_eta: float = 1.0,
    freq_dependence: bool = True,
    freq_alpha_min: float = 0.0,
    freq_alpha_max: float = 4.0,
    freq_alpha_tol: float = 1e-3,
    freq_alpha_max_iter: int = 64,
) -> pd.DataFrame:
    """Detect solar elongation events and annotate rows.

    Model: y(elong, f) = A * exp(-elong / tau) / f^alpha
    """
    out = df.copy()
    out["solar_event_member"] = False
    out["solar_event_year"] = np.nan
    out["solar_event_amp"] = np.nan
    out["solar_event_tau"] = np.nan
    out["solar_event_alpha"] = np.nan
    out["solar_event_delta_chi2"] = np.nan

    if not enabled:
        return out
    if elong_col not in out.columns:
        return out

    t = pd.to_numeric(out[mjd_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(out[resid_col], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(out[sigma_col], errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(out[elong_col], errors="coerce").to_numpy(dtype=float)

    freq = None
    if freq_dependence:
        if freq_col is None or freq_col not in out.columns:
            return out
        freq = pd.to_numeric(out[freq_col], errors="coerce").to_numpy(dtype=float)

    good = np.isfinite(t) & np.isfinite(y) & np.isfinite(s) & (s > 0) & np.isfinite(e)
    if freq_dependence and freq is not None:
        good &= np.isfinite(freq) & (freq != 0)
    if np.count_nonzero(good) < int(min_points_global):
        return out

    w = 1.0 / (s[good] ** 2)
    e_good = e[good]
    y_good = y[good]
    f_good = freq[good] if freq is not None else None

    def _delta_for_alpha(
        alpha: float, ee: np.ndarray, yy: np.ndarray, ww: np.ndarray, ff: np.ndarray | None
    ) -> float:
        if ff is None:
            ff_term = 1.0
        else:
            ff_term = ff**alpha
        tau = _optimize_tau(alpha, ee, yy, ww, ff)
        if not np.isfinite(tau):
            return -np.inf
        model_base = np.exp(-ee / tau) / ff_term
        denom = np.sum(ww * model_base * model_base)
        if denom <= 0:
            return -np.inf
        A = np.sum(ww * model_base * yy) / denom
        if not np.isfinite(A):
            return -np.inf
        chi2_null = np.sum(ww * (yy**2))
        chi2_model = np.sum(ww * ((yy - A * model_base) ** 2))
        return chi2_null - chi2_model

    def _optimize_tau(
        alpha: float, ee: np.ndarray, yy: np.ndarray, ww: np.ndarray, ff: np.ndarray | None
    ) -> float:
        a = float(tau_min_deg)
        b = float(tau_max_deg)
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            return np.nan
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)

        def delta_tau(tau: float) -> float:
            if ff is None:
                ff_term = 1.0
            else:
                ff_term = ff**alpha
            f = np.exp(-ee / tau) / ff_term
            denom = np.sum(ww * f * f)
            if denom <= 0:
                return -np.inf
            A = np.sum(ww * f * yy) / denom
            if not np.isfinite(A):
                return -np.inf
            chi2_null = np.sum(ww * (yy**2))
            chi2_model = np.sum(ww * ((yy - A * f) ** 2))
            return chi2_null - chi2_model

        fc = delta_tau(c)
        fd = delta_tau(d)
        for _ in range(int(freq_alpha_max_iter)):
            if abs(b - a) <= float(freq_alpha_tol):
                break
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = delta_tau(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = delta_tau(d)
        return c if fc >= fd else d

    alpha = 0.0
    if freq_dependence:
        a = float(freq_alpha_min)
        b = float(freq_alpha_max)
        if np.isfinite(a) and np.isfinite(b) and b > a:
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = _delta_for_alpha(c, e_good, y_good, w, f_good)
            fd = _delta_for_alpha(d, e_good, y_good, w, f_good)
            for _ in range(int(freq_alpha_max_iter)):
                if abs(b - a) <= float(freq_alpha_tol):
                    break
                if fc > fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = _delta_for_alpha(c, e_good, y_good, w, f_good)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = _delta_for_alpha(d, e_good, y_good, w, f_good)
            alpha = c if fc >= fd else d

    tau_global = _optimize_tau(alpha, e_good, y_good, w, f_good)
    if not np.isfinite(tau_global):
        return out
    if freq_dependence:
        ff_term = f_good**alpha
    else:
        ff_term = 1.0
    model_base = np.exp(-e_good / tau_global) / ff_term
    denom = np.sum(w * model_base * model_base)
    if denom <= 0:
        return out
    A_global = np.sum(w * model_base * y_good) / denom
    if not np.isfinite(A_global):
        return out
    chi2_null = np.sum(w * (y_good**2))
    chi2_model = np.sum(w * ((y_good - A_global * model_base) ** 2))
    delta_global = chi2_null - chi2_model

    years = _year_from_mjd(t)
    out["solar_event_year"] = years

    for yr in np.unique(years[np.isfinite(years)]):
        mask_year = (years == yr) & good
        if np.count_nonzero(mask_year) < int(min_points_year):
            continue
        near = mask_year & (e <= float(approach_max_deg))
        if np.count_nonzero(near) < int(min_points_near_zero):
            continue

        ee = e[mask_year]
        yy = y[mask_year]
        ss = s[mask_year]
        ww = 1.0 / (ss**2)
        ff = freq[mask_year] if freq is not None else None

        tau_year = _optimize_tau(alpha, ee, yy, ww, ff)
        if not np.isfinite(tau_year):
            continue
        if freq_dependence:
            ff_term = ff**alpha
        else:
            ff_term = 1.0
        base = np.exp(-ee / tau_year) / ff_term
        denom = np.sum(ww * base * base)
        if denom <= 0:
            continue
        A_year = np.sum(ww * base * yy) / denom
        if not np.isfinite(A_year):
            continue
        chi2_null = np.sum(ww * (yy**2))
        chi2_model = np.sum(ww * ((yy - A_year * base) ** 2))
        delta_year = chi2_null - chi2_model

        out.loc[years == yr, "solar_event_amp"] = A_year
        out.loc[years == yr, "solar_event_tau"] = tau_year
        out.loc[years == yr, "solar_event_alpha"] = alpha
        out.loc[years == yr, "solar_event_delta_chi2"] = delta_year

    # Fill remaining years with global values
    missing = out["solar_event_amp"].isna()
    out.loc[missing, "solar_event_amp"] = A_global
    out.loc[missing, "solar_event_tau"] = tau_global
    out.loc[missing, "solar_event_alpha"] = alpha
    out.loc[missing, "solar_event_delta_chi2"] = delta_global

    model = np.exp(-e / out["solar_event_tau"].to_numpy(dtype=float))
    if freq_dependence and freq is not None:
        model = model / (freq**alpha)
    model = model * out["solar_event_amp"].to_numpy(dtype=float)
    z = np.abs(model) / s
    member = np.isfinite(z) & (z >= float(member_eta))
    out["solar_event_member"] = member

    return out
