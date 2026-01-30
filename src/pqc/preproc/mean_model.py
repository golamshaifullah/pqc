"""Mean-model detrending utilities for covariate-conditioned detection."""

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


def fit_binned_mean(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    nbins: int,
    *,
    circular: bool = False,
    min_per_bin: int = 5,
) -> dict:
    """Fit a weighted binned mean model.

    Returns a dict with keys: edges, means, has_mean, circular, min_per_bin.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not np.any(good):
        return {
            "edges": np.array([]),
            "means": np.array([]),
            "has_mean": np.array([]),
            "circular": bool(circular),
            "min_per_bin": int(min_per_bin),
        }

    xg = x[good]
    yg = y[good]
    wg = w[good]

    if circular:
        xg = np.mod(xg, 1.0)

    nbins_req = int(nbins)
    n_points = len(xg)
    nbins_eff = max(3, int(n_points // max(1, int(min_per_bin))))
    nbins_eff = min(nbins_req, nbins_eff) if nbins_req > 0 else nbins_eff

    edges = _bin_edges(xg, nbins_eff, circular)
    if edges.size <= 1:
        return {
            "edges": np.array([]),
            "means": np.array([]),
            "has_mean": np.array([]),
            "circular": bool(circular),
            "min_per_bin": int(min_per_bin),
        }

    nbins_eff = int(edges.size - 1)
    bin_id = np.digitize(xg, edges) - 1
    bin_id = np.clip(bin_id, 0, nbins_eff - 1)

    means = np.full(nbins_eff, np.nan, dtype=float)
    has_mean = np.zeros(nbins_eff, dtype=bool)
    for b in range(nbins_eff):
        m = bin_id == b
        if np.count_nonzero(m) < min_per_bin:
            continue
        wsum = np.sum(wg[m])
        if wsum <= 0:
            continue
        means[b] = np.sum(wg[m] * yg[m]) / wsum
        has_mean[b] = True

    return {
        "edges": edges,
        "means": means,
        "has_mean": has_mean,
        "circular": bool(circular),
        "min_per_bin": int(min_per_bin),
    }


def predict_binned_mean(model: dict, x: np.ndarray) -> np.ndarray:
    """Predict binned mean values for inputs x.

    Missing bins return NaN.
    """
    x = np.asarray(x, dtype=float)
    edges = model.get("edges", np.array([]))
    means = model.get("means", np.array([]))
    has_mean = model.get("has_mean", np.array([]))
    if edges is None or edges.size <= 1 or means is None or means.size == 0:
        return np.full_like(x, np.nan, dtype=float)

    if model.get("circular", False):
        x = np.mod(x, 1.0)

    nbins = int(edges.size - 1)
    bin_id = np.digitize(x, edges) - 1
    bin_id = np.clip(bin_id, 0, nbins - 1)

    yhat = np.full_like(x, np.nan, dtype=float)
    ok = has_mean[bin_id]
    yhat[ok] = means[bin_id[ok]]
    return yhat


def detrend_by_features(
    df: pd.DataFrame,
    features: tuple[str, ...] | list[str],
    *,
    group_cols: tuple[str, ...] = ("group",),
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    nbins_map: dict[str, int] | None = None,
    circular_map: dict[str, bool] | None = None,
    min_per_bin: int = 5,
    out_col: str = "resid_detr",
    store_models: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Detrend residuals by subtracting additive binned mean functions.

    Returns the updated DataFrame and a models dict keyed by (group, feature).
    """
    d = df.copy()
    if not features:
        d[out_col] = d[resid_col]
        return d, {}

    feats = tuple(features)
    nbins_map = nbins_map or {}
    circular_map = circular_map or {}

    for feat in feats:
        if feat not in d.columns:
            d[feat] = np.nan

    models: dict = {}

    groups = [c for c in group_cols if c in d.columns]
    grouped = d.groupby(groups, dropna=False) if groups else [((), d)]

    for key, sub in grouped:
        idx = sub.index
        y = pd.to_numeric(sub[resid_col], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(sub[sigma_col], errors="coerce").to_numpy(dtype=float)
        w = np.zeros_like(s, dtype=float)
        ok = np.isfinite(s) & (s > 0)
        w[ok] = 1.0 / (s[ok] ** 2)

        y_work = y.copy()
        for feat in feats:
            x = pd.to_numeric(sub[feat], errors="coerce").to_numpy(dtype=float)
            nbins = int(nbins_map.get(feat, 12))
            circular = bool(circular_map.get(feat, False))
            model = fit_binned_mean(x, y_work, w, nbins, circular=circular, min_per_bin=min_per_bin)
            yhat = predict_binned_mean(model, x)
            apply = np.isfinite(yhat)
            y_work[apply] = y_work[apply] - yhat[apply]
            if store_models:
                models[(key, feat)] = model

        d.loc[idx, out_col] = y_work

    return d, models
