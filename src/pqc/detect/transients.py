"""Detect exponential-recovery transient events in timing residuals.

We model a transient as:

``y(t) ≈ A * exp(-(t - t0) / tau_rec)`` for ``t >= t0``, evaluated within a
finite window. Candidate ``t0`` values are scanned at observation times and
scored using Δχ² relative to a null model.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def scan_transients(
    df: pd.DataFrame,
    *,
    mjd_col: str = "mjd",
    resid_col: str = "resid",
    sigma_col: str = "sigma",
    exclude_bad_col: str = "bad",
    tau_rec_days: float = 7.0,
    window_mult: float = 5.0,
    min_points: int = 6,
    delta_chi2_thresh: float = 25.0,
    suppress_overlap: bool = True,
) -> pd.DataFrame:
    """Scan for transient exponential recoveries and annotate affected rows.

    Args:
        df: Input DataFrame with timing arrays.
        mjd_col: Column containing MJD values.
        resid_col: Column containing residuals.
        sigma_col: Column containing TOA uncertainties.
        exclude_bad_col: Column marking TOAs to exclude from transient search.
        tau_rec_days: Recovery timescale for the exponential decay (days).
        window_mult: Window length multiplier relative to ``tau_rec_days``.
        min_points: Minimum number of points required in a candidate window.
        delta_chi2_thresh: Minimum Δχ² to accept a candidate.
        suppress_overlap: If True, suppress overlapping transient assignments.

    Returns:
        DataFrame with columns ``transient_id``, ``transient_amp``,
        ``transient_t0``, and ``transient_delta_chi2`` added.

    Notes:
        The algorithm evaluates candidate start times at observation epochs
        only. If ``suppress_overlap`` is enabled, higher-Δχ² events take
        precedence in overlapping windows.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"mjd": [0.0, 1.0, 2.0], "resid": [0.1, 0.1, 0.1], "sigma": [1.0, 1.0, 1.0]})
        >>> out = scan_transients(df, min_points=2, delta_chi2_thresh=0.0)
        >>> "transient_id" in out.columns
        True
    """
    d = df.sort_values(mjd_col).copy()
    d["transient_id"] = -1
    d["transient_amp"] = np.nan
    d["transient_t0"] = np.nan
    d["transient_delta_chi2"] = np.nan

    use = np.ones(len(d), dtype=bool)
    if exclude_bad_col in d.columns:
        use &= ~d[exclude_bad_col].fillna(False).to_numpy()

    t = d[mjd_col].to_numpy(dtype=float)
    y = d[resid_col].to_numpy(dtype=float)
    s = d[sigma_col].to_numpy(dtype=float)

    cand = np.where(use)[0]
    if len(cand) < min_points:
        return d

    w_end = window_mult * tau_rec_days
    events = []

    for idx0 in cand:
        t0 = t[idx0]
        in_win = use & (t >= t0) & (t <= t0 + w_end)
        if np.count_nonzero(in_win) < min_points:
            continue

        tt = t[in_win] - t0
        yy = y[in_win]
        ww = 1.0 / (s[in_win] ** 2)

        f = np.exp(-tt / tau_rec_days)
        denom = np.sum(ww * f * f)
        if denom <= 0:
            continue

        A = np.sum(ww * f * yy) / denom

        chi2_null = np.sum(ww * (yy ** 2))
        chi2_model = np.sum(ww * ((yy - A * f) ** 2))
        delta = chi2_null - chi2_model

        if delta >= delta_chi2_thresh:
            events.append((t0, A, delta, in_win.copy()))

    if not events:
        return d

    events.sort(key=lambda e: e[2], reverse=True)

    assigned = np.zeros(len(d), dtype=bool)
    kept = []

    for t0, A, delta, in_win in events:
        if suppress_overlap and np.any(assigned & in_win):
            continue
        kept.append((t0, A, delta, in_win))
        assigned |= in_win

    for k, (t0, A, delta, in_win) in enumerate(kept):
        d.loc[in_win, "transient_id"] = k
        d.loc[in_win, "transient_amp"] = A
        d.loc[in_win, "transient_t0"] = t0
        d.loc[in_win, "transient_delta_chi2"] = delta

    return d
