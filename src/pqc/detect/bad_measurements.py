"""Detect bad measurements via OU innovations and day-level FDR.

This module computes normalized innovations from an OU process and uses
Benjamini–Hochberg false discovery rate control on day-level extremes to flag
bad measurements.

See Also:
    pqc.detect.ou.ou_innovations_z: OU innovations used in this detector.
    pqc.utils.stats.bh_fdr: Benjamini–Hochberg FDR control.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pqc.detect.ou import estimate_q, ou_innovations_z
from pqc.utils.stats import norm_abs_sf, bh_fdr


def detect_bad(
    df: pd.DataFrame,
    *,
    mjd_col: str = "mjd",
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    day_col: str = "day",
    tau_corr_days: float = 0.02,
    fdr_q: float = 0.01,
    mark_only_worst_per_day: bool = True,
) -> pd.DataFrame:
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
        df (pandas.DataFrame): Input DataFrame with timing arrays.
        mjd_col (str): Column containing MJD values.
        resid_col (str): Column containing residuals.
        sigma_col (str): Column containing TOA uncertainties.
        day_col (str): Column containing integer MJD day labels.
        tau_corr_days (float): OU correlation timescale in days.
        fdr_q (float): Target FDR rate for day-level tests.
        mark_only_worst_per_day (bool): If True, mark only the worst TOA per
            bad day.

    Returns:
        pandas.DataFrame: Copy with added columns ``z``, ``_q_hat``,
        ``bad_day``, and ``bad``.

    Notes:
        This function expects ``df`` to contain finite values in ``mjd_col``,
        ``resid_col``, and ``sigma_col``. Rows with invalid innovations are
        preserved but will not be considered for day-level testing.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"mjd": [1.0, 1.5], "resid": [0.1, -0.2], "sigma": [1.0, 1.0], "day": [1, 1]})
        >>> out = detect_bad(df)
        >>> set(["bad", "bad_day", "z"]).issubset(out.columns)
        True
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
