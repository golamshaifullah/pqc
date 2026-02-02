"""Detect global glitch events in timing residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def scan_glitches(
    df: pd.DataFrame,
    *,
    mjd_col: str = "mjd",
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    enabled: bool = True,
    min_points: int = 20,
    delta_chi2_thresh: float = 25.0,
    suppress_overlap: bool = True,
    member_eta: float = 1.0,
    peak_tau_days: float = 30.0,
) -> pd.DataFrame:
    """Scan for glitch-like events (step + linear ramp, or peak + linear ramp)."""
    out = df.sort_values(mjd_col).copy()
    out["glitch_id"] = -1
    out["glitch_t0"] = np.nan
    out["glitch_amp"] = np.nan
    out["glitch_slope"] = np.nan
    out["glitch_peak_tau_days"] = np.nan
    out["glitch_model"] = ""
    out["glitch_delta_chi2"] = np.nan
    out["glitch_member"] = False

    if not enabled:
        return out

    t = pd.to_numeric(out[mjd_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(out[resid_col], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(out[sigma_col], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(t) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    exclude = np.zeros(len(out), dtype=bool)
    if "exp_dip_member" in out.columns:
        exclude |= out["exp_dip_member"].fillna(False).to_numpy()
    if "exp_dip_global_id" in out.columns:
        exclude |= out["exp_dip_global_id"].fillna(-1).to_numpy() >= 0
    good &= ~exclude
    if np.count_nonzero(good) < int(min_points):
        return out

    candidates = np.where(good)[0]
    events: list[tuple[float, float, float, float, str, float, np.ndarray]] = []

    for idx0 in candidates:
        t0 = t[idx0]
        in_win = good & (t >= t0)
        if np.count_nonzero(in_win) < int(min_points):
            continue

        tt = t[in_win] - t0
        yy = y[in_win]
        ww = 1.0 / (s[in_win] ** 2)

        # Model 1: step + linear ramp: A + B*dt
        X1 = np.vstack([np.ones_like(tt), tt]).T
        W = np.sqrt(ww)
        Xw = X1 * W[:, None]
        yw = yy * W
        try:
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except Exception:
            beta = np.array([np.nan, np.nan])
        A1, B1 = beta
        if np.isfinite(A1) and np.isfinite(B1):
            model1 = X1 @ beta
            chi2_null = np.sum(ww * (yy ** 2))
            chi2_model = np.sum(ww * ((yy - model1) ** 2))
            delta1 = chi2_null - chi2_model
        else:
            delta1 = -np.inf

        # Model 2: peak + linear ramp: A*exp(-dt/tau) + B*dt
        tau = float(peak_tau_days)
        f = np.exp(-tt / tau)
        X2 = np.vstack([f, tt]).T
        Xw2 = X2 * W[:, None]
        try:
            beta2, *_ = np.linalg.lstsq(Xw2, yw, rcond=None)
        except Exception:
            beta2 = np.array([np.nan, np.nan])
        A2, B2 = beta2
        if np.isfinite(A2) and np.isfinite(B2):
            model2 = X2 @ beta2
            chi2_null = np.sum(ww * (yy ** 2))
            chi2_model = np.sum(ww * ((yy - model2) ** 2))
            delta2 = chi2_null - chi2_model
        else:
            delta2 = -np.inf

        if max(delta1, delta2) < float(delta_chi2_thresh):
            continue

        if delta2 > delta1:
            events.append((t0, A2, B2, tau, "peak_ramp", delta2, in_win.copy()))
        else:
            events.append((t0, A1, B1, np.nan, "step_ramp", delta1, in_win.copy()))

    if not events:
        return out

    events.sort(key=lambda e: e[5], reverse=True)
    assigned = np.zeros(len(out), dtype=bool)
    kept = []
    for t0, A, B, tau, kind, delta, in_win in events:
        if suppress_overlap and np.any(assigned & in_win):
            continue
        kept.append((t0, A, B, tau, kind, delta, in_win))
        assigned |= in_win

    for k, (t0, A, B, tau, kind, delta, in_win) in enumerate(kept):
        tt = t[in_win] - t0
        if kind == "peak_ramp":
            model = A * np.exp(-tt / tau) + B * tt
        else:
            model = A + B * tt
        z = np.abs(model) / s[in_win]
        member = np.zeros_like(in_win, dtype=bool)
        member[in_win] = np.isfinite(z) & (z >= float(member_eta))
        if np.any(exclude):
            member[exclude] = False

        out.loc[member, "glitch_id"] = k
        out.loc[member, "glitch_t0"] = t0
        out.loc[member, "glitch_amp"] = A
        out.loc[member, "glitch_slope"] = B
        out.loc[member, "glitch_peak_tau_days"] = tau
        out.loc[member, "glitch_model"] = kind
        out.loc[member, "glitch_delta_chi2"] = delta
        out.loc[member, "glitch_member"] = True

    return out
