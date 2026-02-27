"""Detect orbital-phase eclipse events in timing residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_eclipse_events(
    df: pd.DataFrame,
    *,
    phase_col: str = "orbital_phase",
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    freq_col: str | None = "freq",
    enabled: bool = True,
    center_phase: float = 0.25,
    min_points: int = 30,
    width_min: float = 0.01,
    width_max: float = 0.5,
    member_eta: float = 1.0,
    freq_dependence: bool = True,
    freq_alpha_min: float = 0.0,
    freq_alpha_max: float = 4.0,
    freq_alpha_tol: float = 1e-3,
    freq_alpha_max_iter: int = 64,
) -> pd.DataFrame:
    """Detect eclipse events and annotate rows.

    Model: y(phase, f) = A * exp(-d / width) / f^alpha
    where d is distance to eclipse center in phase folded to [0, 0.5].
    """
    out = df.copy()
    out["eclipse_event_member"] = False
    out["eclipse_event_amp"] = np.nan
    out["eclipse_event_width"] = np.nan
    out["eclipse_event_alpha"] = np.nan
    out["eclipse_event_delta_chi2"] = np.nan

    if not enabled or phase_col not in out.columns:
        return out

    phase = pd.to_numeric(out[phase_col], errors="coerce").to_numpy(dtype=float)
    resid = pd.to_numeric(out[resid_col], errors="coerce").to_numpy(dtype=float)
    sigma = pd.to_numeric(out[sigma_col], errors="coerce").to_numpy(dtype=float)
    freq = None
    if freq_dependence:
        if freq_col is None or freq_col not in out.columns:
            return out
        freq = pd.to_numeric(out[freq_col], errors="coerce").to_numpy(dtype=float)

    good = np.isfinite(phase) & np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
    if freq_dependence and freq is not None:
        good &= np.isfinite(freq) & (freq != 0)
    if np.count_nonzero(good) < int(min_points):
        return out

    d = np.minimum(np.abs(phase - center_phase), 1.0 - np.abs(phase - center_phase))
    d_good = d[good]
    y = resid[good]
    w = 1.0 / (sigma[good] ** 2)
    f_good = freq[good] if freq is not None else None

    def _optimize_width(alpha: float) -> float:
        a = float(width_min)
        b = float(width_max)
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            return np.nan
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d2 = a + gr * (b - a)

        def delta_width(width: float) -> float:
            if f_good is None:
                ff_term = 1.0
            else:
                ff_term = f_good**alpha
            model_base = np.exp(-d_good / width) / ff_term
            denom = np.sum(w * model_base * model_base)
            if denom <= 0:
                return -np.inf
            A = np.sum(w * model_base * y) / denom
            if not np.isfinite(A):
                return -np.inf
            chi2_null = np.sum(w * (y**2))
            chi2_model = np.sum(w * ((y - A * model_base) ** 2))
            return chi2_null - chi2_model

        fc = delta_width(c)
        fd = delta_width(d2)
        for _ in range(int(freq_alpha_max_iter)):
            if abs(b - a) <= float(freq_alpha_tol):
                break
            if fc > fd:
                b = d2
                d2 = c
                fd = fc
                c = b - gr * (b - a)
                fc = delta_width(c)
            else:
                a = c
                c = d2
                fc = fd
                d2 = a + gr * (b - a)
                fd = delta_width(d2)
        return c if fc >= fd else d2

    def _delta_for_alpha(alpha: float) -> float:
        width = _optimize_width(alpha)
        if not np.isfinite(width):
            return -np.inf
        if f_good is None:
            ff_term = 1.0
        else:
            ff_term = f_good**alpha
        model_base = np.exp(-d_good / width) / ff_term
        denom = np.sum(w * model_base * model_base)
        if denom <= 0:
            return -np.inf
        A = np.sum(w * model_base * y) / denom
        if not np.isfinite(A):
            return -np.inf
        chi2_null = np.sum(w * (y**2))
        chi2_model = np.sum(w * ((y - A * model_base) ** 2))
        return chi2_null - chi2_model

    alpha = 0.0
    if freq_dependence:
        a = float(freq_alpha_min)
        b = float(freq_alpha_max)
        if np.isfinite(a) and np.isfinite(b) and b > a:
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d2 = a + gr * (b - a)
            fc = _delta_for_alpha(c)
            fd = _delta_for_alpha(d2)
            for _ in range(int(freq_alpha_max_iter)):
                if abs(b - a) <= float(freq_alpha_tol):
                    break
                if fc > fd:
                    b = d2
                    d2 = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = _delta_for_alpha(c)
                else:
                    a = c
                    c = d2
                    fc = fd
                    d2 = a + gr * (b - a)
                    fd = _delta_for_alpha(d2)
            alpha = c if fc >= fd else d2

    width = _optimize_width(alpha)
    if not np.isfinite(width):
        return out
    if f_good is None:
        ff_term = 1.0
    else:
        ff_term = f_good**alpha
    base = np.exp(-d_good / width) / ff_term
    denom = np.sum(w * base * base)
    if denom <= 0:
        return out
    A = np.sum(w * base * y) / denom
    if not np.isfinite(A):
        return out
    chi2_null = np.sum(w * (y**2))
    chi2_model = np.sum(w * ((y - A * base) ** 2))
    delta = chi2_null - chi2_model

    out["eclipse_event_amp"] = A
    out["eclipse_event_width"] = width
    out["eclipse_event_alpha"] = alpha
    out["eclipse_event_delta_chi2"] = delta

    model = np.exp(-d / width)
    if freq_dependence and freq is not None:
        model = model / (freq**alpha)
    model = model * A
    z = np.abs(model) / sigma
    out["eclipse_event_member"] = np.isfinite(z) & (z >= float(member_eta))
    return out
