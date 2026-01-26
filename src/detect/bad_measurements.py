"""Detect bad measurements via OU innovations and day-level FDR."""

from __future__ import annotations
import numpy as np
import pandas as pd
from pqc.detect.ou import estimate_q, ou_innovations_z
from pqc.utils.stats import norm_abs_sf, bh_fdr

def detect_bad(df: pd.DataFrame, *,
               mjd_col="mjd", resid_col="resid", sigma_col="sigma", day_col="day",
               tau_corr_days=0.02, fdr_q=0.01, mark_only_worst_per_day=True) -> pd.DataFrame:
    """Flag bad measurements using OU innovations and BH-FDR on days.

    The method assumes:
        - Residuals are Gaussian with short-term OU correlation.
        - Bad measurements are uncorrelated across observing days.

    Steps:
        1. Compute OU innovations ``z`` for the time series.
        2. Aggregate by day using max |z|.
        3. Apply BH-FDR to day-level p-values.
        4. Mark either the worst TOA per bad day or all TOAs on that day.

    Args:
        df: Input DataFrame with timing arrays.
        mjd_col: Column containing MJD values.
        resid_col: Column containing residuals.
        sigma_col: Column containing TOA uncertainties.
        day_col: Column containing integer MJD day labels.
        tau_corr_days: OU correlation timescale in days.
        fdr_q: Target FDR rate for day-level tests.
        mark_only_worst_per_day: If True, mark only the worst TOA per bad day.

    Returns:
        DataFrame with added columns ``z``, ``_q_hat``, ``bad_day``, and ``bad``.
    """
    d = df.sort_values(mjd_col).copy()

    t = d[mjd_col].to_numpy(dtype=float)
    y = d[resid_col].to_numpy(dtype=float)
    s = d[sigma_col].to_numpy(dtype=float)

    q_hat = estimate_q(t, y, s, tau_corr_days)
    z = ou_innovations_z(t, y, s, tau_corr_days, q_hat)

    d["z"] = z
    d["_q_hat"] = float(q_hat)
    d["bad_day"] = False
    d["bad"] = False

    good = np.isfinite(z)
    if not np.any(good):
        return d

    tmp = d.loc[good, [day_col, "z"]].copy()
    tmp["absz"] = tmp["z"].abs()
    day_max = tmp.groupby(day_col)["absz"].max()

    pvals = norm_abs_sf(day_max.to_numpy())
    is_bad_day = bh_fdr(pvals, q=fdr_q)
    bad_days = day_max.index.to_numpy()[is_bad_day]

    if len(bad_days) == 0:
        return d

    d.loc[d[day_col].isin(bad_days), "bad_day"] = True

    if mark_only_worst_per_day:
        for dd in bad_days:
            idx = d.loc[d[day_col] == dd, "z"].abs().idxmax()
            d.loc[idx, "bad"] = True
    else:
        d.loc[d["bad_day"], "bad"] = True

    return d
