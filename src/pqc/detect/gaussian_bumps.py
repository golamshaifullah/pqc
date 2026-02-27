"""Detect global Gaussian-bump events in timing residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def scan_gaussian_bumps(
    df: pd.DataFrame,
    *,
    mjd_col: str = "mjd",
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    freq_col: str | None = "freq",
    enabled: bool = True,
    min_duration_days: float = 60.0,
    max_duration_days: float = 1500.0,
    n_durations: int = 6,
    min_points: int = 20,
    delta_chi2_thresh: float = 25.0,
    suppress_overlap: bool = True,
    member_eta: float = 1.0,
    freq_dependence: bool = True,
    freq_alpha_min: float = 0.0,
    freq_alpha_max: float = 4.0,
    freq_alpha_tol: float = 1e-3,
    freq_alpha_max_iter: int = 64,
) -> pd.DataFrame:
    """Scan for global Gaussian-bump events and annotate affected rows.

    Models compared (amplitude fitted per candidate):
        - gaussian: exp(-0.5 * ((t-t0)/sigma)^2)
        - laplace: exp(-|t-t0|/tau)
        - plateau: exp(-|t-t0|/tau) with a flat top of width L/3
    """
    out = df.sort_values(mjd_col).copy()
    out["gaussian_bump_id"] = -1
    out["gaussian_bump_amp"] = np.nan
    out["gaussian_bump_t0"] = np.nan
    out["gaussian_bump_duration_days"] = np.nan
    out["gaussian_bump_model"] = ""
    out["gaussian_bump_alpha"] = np.nan
    out["gaussian_bump_delta_chi2"] = np.nan
    out["gaussian_bump_member"] = False

    if not enabled:
        return out

    t = pd.to_numeric(out[mjd_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(out[resid_col], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(out[sigma_col], errors="coerce").to_numpy(dtype=float)
    freq = None
    if freq_dependence:
        if freq_col is None or freq_col not in out.columns:
            return out
        freq = pd.to_numeric(out[freq_col], errors="coerce").to_numpy(dtype=float)

    good = np.isfinite(t) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    if freq_dependence and freq is not None:
        good &= np.isfinite(freq) & (freq != 0)
    exclude = np.zeros(len(out), dtype=bool)
    if "exp_dip_member" in out.columns:
        exclude |= out["exp_dip_member"].fillna(False).to_numpy()
    if "exp_dip_global_id" in out.columns:
        exclude |= out["exp_dip_global_id"].fillna(-1).to_numpy() >= 0
    if "glitch_member" in out.columns:
        exclude |= out["glitch_member"].fillna(False).to_numpy()
    if "glitch_id" in out.columns:
        exclude |= out["glitch_id"].fillna(-1).to_numpy() >= 0
    good &= ~exclude
    if np.count_nonzero(good) < int(min_points):
        return out

    t_good = t[good]

    durations = np.linspace(float(min_duration_days), float(max_duration_days), int(n_durations))
    models = ("gaussian", "laplace", "plateau")

    def _shape(tt: np.ndarray, t0: float, L: float, kind: str) -> np.ndarray:
        if kind == "gaussian":
            sigma = L / 6.0
            return np.exp(-0.5 * ((tt - t0) / sigma) ** 2)
        if kind == "laplace":
            tau = L / 6.0
            return np.exp(-np.abs(tt - t0) / tau)
        # plateau: flat top of width L/3 with exponential shoulders
        tau = L / 10.0
        half = L / 6.0
        d = np.abs(tt - t0)
        out = np.ones_like(tt, dtype=float)
        shoulder = d > half
        out[shoulder] = np.exp(-(d[shoulder] - half) / tau)
        return out

    def _delta_for_alpha(
        alpha: float,
        tt: np.ndarray,
        yy: np.ndarray,
        ww: np.ndarray,
        ff: np.ndarray | None,
        base: np.ndarray,
    ) -> float:
        if ff is None:
            model = base
        else:
            model = base / (ff**alpha)
        denom = np.sum(ww * model * model)
        if denom <= 0:
            return -np.inf
        A = np.sum(ww * model * yy) / denom
        if not np.isfinite(A):
            return -np.inf
        chi2_null = np.sum(ww * (yy**2))
        chi2_model = np.sum(ww * ((yy - A * model) ** 2))
        return chi2_null - chi2_model

    def _optimize_alpha(
        tt: np.ndarray, yy: np.ndarray, ww: np.ndarray, ff: np.ndarray | None, base: np.ndarray
    ) -> tuple[float, float]:
        a = float(freq_alpha_min)
        b = float(freq_alpha_max)
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            return np.nan, -np.inf
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _delta_for_alpha(c, tt, yy, ww, ff, base)
        fd = _delta_for_alpha(d, tt, yy, ww, ff, base)
        for _ in range(int(freq_alpha_max_iter)):
            if abs(b - a) <= float(freq_alpha_tol):
                break
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = _delta_for_alpha(c, tt, yy, ww, ff, base)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = _delta_for_alpha(d, tt, yy, ww, ff, base)
        if fc >= fd:
            return c, fc
        return d, fd

    events: list[tuple[float, float, str, float, float, float, np.ndarray]] = []
    for t0 in t_good:
        for L in durations:
            t_start = t0 - L / 2.0
            t_end = t0 + L / 2.0
            in_win = good & (t >= t_start) & (t <= t_end)
            if np.count_nonzero(in_win) < int(min_points):
                continue
            tt = t[in_win]
            yy = y[in_win]
            ww = 1.0 / (s[in_win] ** 2)
            ff = freq[in_win] if freq is not None else None

            best = None
            for kind in models:
                base = _shape(tt, t0, L, kind)
                if freq_dependence:
                    alpha, delta = _optimize_alpha(tt, yy, ww, ff, base)
                else:
                    alpha = np.nan
                    delta = _delta_for_alpha(0.0, tt, yy, ww, None, base)
                if not np.isfinite(delta):
                    continue
                if delta >= float(delta_chi2_thresh):
                    denom = (
                        np.sum(ww * base * base)
                        if ff is None
                        else np.sum(ww * (base / (ff**alpha)) ** 2)
                    )
                    if denom <= 0:
                        continue
                    model = base if ff is None else base / (ff**alpha)
                    A = np.sum(ww * model * yy) / denom
                    if not np.isfinite(A):
                        continue
                    if best is None or delta > best[2]:
                        best = (A, delta, alpha, kind)
            if best is None:
                continue
            A, delta, alpha, kind = best
            events.append((t0, L, kind, A, alpha, delta, in_win.copy()))

    if not events:
        return out

    events.sort(key=lambda e: e[5], reverse=True)
    assigned = np.zeros(len(out), dtype=bool)
    kept = []
    for t0, L, kind, A, alpha, delta, in_win in events:
        if suppress_overlap and np.any(assigned & in_win):
            continue
        kept.append((t0, L, kind, A, alpha, delta, in_win))
        assigned |= in_win

    for k, (t0, L, kind, A, alpha, delta, in_win) in enumerate(kept):
        tt = t[in_win]
        base = _shape(tt, t0, L, kind)
        if freq_dependence and freq is not None:
            model = base / (freq[in_win] ** alpha)
        else:
            model = base
        model = model * A
        z = np.abs(model) / s[in_win]
        member = np.zeros_like(in_win, dtype=bool)
        member[in_win] = np.isfinite(z) & (z >= float(member_eta))

        out.loc[member, "gaussian_bump_id"] = k
        out.loc[member, "gaussian_bump_amp"] = A
        out.loc[member, "gaussian_bump_t0"] = t0
        out.loc[member, "gaussian_bump_duration_days"] = L
        out.loc[member, "gaussian_bump_model"] = kind
        out.loc[member, "gaussian_bump_alpha"] = alpha
        out.loc[member, "gaussian_bump_delta_chi2"] = delta
        out.loc[member, "gaussian_bump_member"] = True

    return out
