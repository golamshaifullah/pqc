"""Robust outlier detection using MAD-based z-scores."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_robust_outliers(
    df: pd.DataFrame,
    *,
    resid_col: str = "resid",
    sigma_col: str | None = None,
    z_thresh: float = 5.0,
    prefix: str = "robust",
) -> pd.DataFrame:
    """Flag robust outliers using median/MAD.

    Adds columns: <prefix>_z, <prefix>_outlier
    """
    out = df.copy()
    out[f"{prefix}_z"] = np.nan
    out[f"{prefix}_outlier"] = False

    if resid_col not in out.columns:
        return out

    y = pd.to_numeric(out[resid_col], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(y)
    if not np.any(good):
        return out

    med = np.nanmedian(y[good])
    mad = np.nanmedian(np.abs(y[good] - med))
    if not np.isfinite(mad) or mad == 0:
        return out

    z = 0.6745 * (y - med) / mad
    out.loc[good, f"{prefix}_z"] = z[good]
    out.loc[good, f"{prefix}_outlier"] = np.abs(z[good]) >= float(z_thresh)
    return out
