"""Detect feature-domain structure and provide simple detrending utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pqc.utils.stats import chi2_sf_approx

def _bin_edges(x: np.ndarray, nbins: int, circular: bool) -> np.ndarray:
    if circular:
        return np.linspace(0.0, 1.0, nbins + 1)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.array([lo, hi]) if np.isfinite(lo) else np.array([0.0, 1.0])
    return np.linspace(lo, hi, nbins + 1)

def detect_binned_structure(
    df: pd.DataFrame,
    feature_col: str,
    *,
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    nbins: int = 12,
    circular: bool = False,
    min_per_bin: int = 3,
) -> dict:
    """Test for coherent feature dependence using binned weighted means."""
    d = df[[feature_col, resid_col, sigma_col]].dropna().copy()
    if d.empty:
        return {"chi2": np.nan, "dof": 0, "p_like": np.nan, "bin_table": pd.DataFrame()}

    x = d[feature_col].to_numpy(dtype=float)
    y = d[resid_col].to_numpy(dtype=float)
    s = d[sigma_col].to_numpy(dtype=float)

    if circular:
        x = np.mod(x, 1.0)

    edges = _bin_edges(x, nbins, circular)
    if edges.size <= 1:
        return {"chi2": np.nan, "dof": 0, "p_like": np.nan, "bin_table": pd.DataFrame()}

    bin_id = np.digitize(x, edges) - 1
    bin_id = np.clip(bin_id, 0, nbins - 1)

    rows = []
    chi2 = 0.0
    dof = 0
    for b in range(nbins):
        m = bin_id == b
        if np.count_nonzero(m) < min_per_bin:
            continue
        w = 1.0 / (s[m] ** 2)
        wsum = np.sum(w)
        if wsum <= 0:
            continue
        mu = np.sum(w * y[m]) / wsum
        mu_err = np.sqrt(1.0 / wsum)
        chi2 += (mu / mu_err) ** 2
        dof += 1
        rows.append(
            {
                "bin": b,
                "mean": mu,
                "mean_err": mu_err,
                "n": int(np.count_nonzero(m)),
                "edge_lo": float(edges[b]),
                "edge_hi": float(edges[b + 1]),
            }
        )

    p_like = chi2_sf_approx(chi2, dof) if dof > 0 else np.nan
    return {"chi2": chi2, "dof": dof, "p_like": p_like, "bin_table": pd.DataFrame(rows)}

def detrend_residuals_binned(
    df: pd.DataFrame,
    feature_col: str,
    *,
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    nbins: int = 12,
    circular: bool = False,
    min_per_bin: int = 3,
    out_col: str = "resid_detrended",
) -> pd.DataFrame:
    """Subtract a binned weighted-mean trend from residuals."""
    d = df.copy()
    if feature_col not in d.columns:
        d[out_col] = d[resid_col]
        return d

    x = pd.to_numeric(d[feature_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(d[resid_col], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(d[sigma_col], errors="coerce").to_numpy(dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    d[out_col] = y.copy()
    if not np.any(good):
        return d

    if circular:
        x = np.mod(x, 1.0)

    edges = _bin_edges(x[good], nbins, circular)
    if edges.size <= 1:
        return d

    bin_id = np.digitize(x, edges) - 1
    bin_id = np.clip(bin_id, 0, nbins - 1)

    means = np.zeros(nbins, dtype=float)
    has_mean = np.zeros(nbins, dtype=bool)
    for b in range(nbins):
        m = good & (bin_id == b)
        if np.count_nonzero(m) < min_per_bin:
            continue
        w = 1.0 / (s[m] ** 2)
        wsum = np.sum(w)
        if wsum <= 0:
            continue
        means[b] = np.sum(w * y[m]) / wsum
        has_mean[b] = True

    apply = good & has_mean[bin_id]
    d.loc[apply, out_col] = y[apply] - means[bin_id[apply]]
    return d
