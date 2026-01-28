"""Variance stabilization utilities for residual preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _bin_edges(x: np.ndarray, nbins: int, circular: bool) -> np.ndarray:
    if circular:
        return np.linspace(0.0, 1.0, nbins + 1)
    lo = float(np.nanmin(x)) if x.size else np.nan
    hi = float(np.nanmax(x)) if x.size else np.nan
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.array([lo, hi]) if np.isfinite(lo) else np.array([0.0, 1.0])
    return np.linspace(lo, hi, nbins + 1)


def _robust_sigma(y: np.ndarray) -> float:
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    if not np.isfinite(mad) or mad <= 0:
        return np.nan
    return 1.4826 * mad


def rescale_by_feature(
    df: pd.DataFrame,
    feature: str,
    *,
    group_cols: tuple[str, ...] = ("group",),
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    nbins: int = 12,
    circular: bool = False,
    min_per_bin: int = 5,
    out_resid_col: str = "resid_proc",
    out_sigma_col: str = "sigma_proc",
) -> pd.DataFrame:
    """Rescale residuals and sigmas using a binned robust scale vs feature."""
    d = df.copy()
    if feature not in d.columns:
        d[out_resid_col] = d[resid_col]
        d[out_sigma_col] = d[sigma_col]
        return d

    for col in (resid_col, sigma_col):
        if col not in d.columns:
            d[out_resid_col] = d.get(resid_col, np.nan)
            d[out_sigma_col] = d.get(sigma_col, np.nan)
            return d

    groups = [c for c in group_cols if c in d.columns]
    grouped = d.groupby(groups, dropna=False) if groups else [((), d)]

    for key, sub in grouped:
        idx = sub.index
        x = pd.to_numeric(sub[feature], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[resid_col], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(sub[sigma_col], errors="coerce").to_numpy(dtype=float)

        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
        if not np.any(good):
            d.loc[idx, out_resid_col] = y
            d.loc[idx, out_sigma_col] = s
            continue

        xg = x[good]
        if circular:
            xg = np.mod(xg, 1.0)

        nbins_req = int(nbins)
        n_points = len(xg)
        nbins_eff = max(3, int(n_points // max(1, int(min_per_bin))))
        nbins_eff = min(nbins_req, nbins_eff) if nbins_req > 0 else nbins_eff

        edges = _bin_edges(xg, nbins_eff, circular)
        if edges.size <= 1:
            d.loc[idx, out_resid_col] = y
            d.loc[idx, out_sigma_col] = s
            continue

        nbins_eff = int(edges.size - 1)
        bin_id = np.digitize(xg, edges) - 1
        bin_id = np.clip(bin_id, 0, nbins_eff - 1)

        scales = np.full(nbins_eff, np.nan, dtype=float)
        has_scale = np.zeros(nbins_eff, dtype=bool)
        for b in range(nbins_eff):
            m = bin_id == b
            if np.count_nonzero(m) < min_per_bin:
                continue
            sigma_hat = _robust_sigma(y[good][m])
            if np.isfinite(sigma_hat) and sigma_hat > 0:
                scales[b] = sigma_hat
                has_scale[b] = True

        x_all = x.copy()
        if circular:
            x_all = np.mod(x_all, 1.0)
        bin_all = np.digitize(x_all, edges) - 1
        bin_all = np.clip(bin_all, 0, nbins_eff - 1)
        scale_all = np.ones_like(y, dtype=float)
        apply = has_scale[bin_all]
        scale_all[apply] = scales[bin_all[apply]]

        y_scaled = y / scale_all
        s_scaled = s / scale_all

        d.loc[idx, out_resid_col] = y_scaled
        d.loc[idx, out_sigma_col] = s_scaled

    return d
