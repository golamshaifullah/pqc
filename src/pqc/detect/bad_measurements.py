"""Detect bad measurements using OU innovations and day-level FDR control.

This module implements a two-stage statistical procedure:

1. **Model short-timescale correlated residuals** with an
   Ornstein–Uhlenbeck (OU) process and compute normalized innovations
   :math:`z_i`.
2. **Control multiplicity across days** by converting daily maxima of
   :math:`|z_i|` into p-values and applying Benjamini–Hochberg false
   discovery rate (FDR) control.

The OU step is used because pulsar timing residuals can be locally correlated;
innovation whitening makes single-point surprises more interpretable. The
day-level FDR step is used to reduce false positives under repeated testing.

Notes
-----
Definition (OU innovations)
    For irregularly sampled times :math:`t_i`, residuals :math:`y_i`, and
    correlation timescale :math:`\\tau`, an OU predictor uses
    :math:`\\phi_i = \\exp(-(t_i-t_{i-1})/\\tau)`. The innovation is
    :math:`e_i = y_i - \\phi_i y_{i-1}` and the normalized innovation is
    :math:`z_i = e_i / \\sqrt{\\mathrm{Var}(e_i)}`.

Definition (FDR)
    FDR is :math:`\\mathbb{E}[V/\\max(R,1)]`, where :math:`V` is number of
    false rejections and :math:`R` is total rejections.

Assumptions
    - Innovations are approximately Gaussian under the null model.
    - Day-level tests are independent or positively dependent (BH validity).
    - A daily max-:math:`|z|` statistic captures day-level contamination.

Interpretation
    A flagged day indicates at least one measurement is unusually inconsistent
    with OU-consistent noise at the chosen FDR level.

Caveats
    - Misspecified :math:`\\tau` or variance can distort p-values.
    - Daily aggregation can lose within-day structure.
    - Heavy tails can inflate false positives if Gaussian tails are assumed.

Worked Example
--------------
Suppose day-level p-values are ``[0.001, 0.012, 0.08, 0.2]`` and
``fdr_q = 0.02``. BH thresholds for sorted p-values are
``[0.005, 0.01, 0.015, 0.02]``. Only ``0.001`` is below threshold, so one day
is flagged.

References
----------
.. [1] Uhlenbeck, G. E., & Ornstein, L. S. (1930), *Physical Review*, 36, 823.
.. [2] Benjamini, Y., & Hochberg, Y. (1995), *JRSS B*, 57(1), 289-300.
.. [3] Brockwell, P. J., & Davis, R. A. (2002), *Introduction to Time Series
   and Forecasting*, 2nd ed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pqc.detect.ou import estimate_q, ou_innovations_z
from pqc.utils.stats import bh_fdr, norm_abs_sf


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

    Raises:
        KeyError
            If any required columns are missing from ``df``.
        ValueError
            If numeric coercion yields invalid arrays for core computations.

    Notes:
        **Statistic and formula**
            Let :math:`z_i` be normalized OU innovations.
            For each day :math:`d`, compute
            :math:`T_d = \\max_{i \\in d}|z_i|`.
            Under a Gaussian-tail approximation, the two-sided p-value is
            :math:`p_d = 2(1-\\Phi(T_d))`.
            BH is applied to :math:`p_d` at level ``fdr_q``.

        **Why used here**
            Day-level FDR gives operationally robust triage while controlling
            expected false discovery fraction across many days.

        **Interpretation**
            ``bad_day=True`` marks days rejected by BH; ``bad=True`` marks
            either the worst TOA per bad day or all TOAs in each bad day,
            controlled by ``mark_only_worst_per_day``.

        **Caveats**
            If noise is strongly non-Gaussian or day definitions are poor,
            p-values may be miscalibrated.

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
