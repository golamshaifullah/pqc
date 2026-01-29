"""Detect exponential-recovery transient events in timing residuals.

We model a transient as:

``y(t) ≈ A * exp(-(t - t0) / tau_rec)`` for ``t >= t0``, evaluated within a
finite window. Candidate ``t0`` values are scanned at observation times and
scored using Δχ² relative to a null model.

See Also:
    pqc.detect.bad_measurements.detect_bad: Bad measurement detector.
    pqc.utils.stats: Statistical helpers used across detectors.
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
    member_eta: float = 1.0,
    instrument: bool = False,
) -> pd.DataFrame:
    """Scan for transient exponential recoveries and annotate affected rows.

    Args:
        df (pandas.DataFrame): Input DataFrame with timing arrays.
        mjd_col (str): Column containing MJD values.
        resid_col (str): Column containing residuals.
        sigma_col (str): Column containing TOA uncertainties.
        exclude_bad_col (str): Column marking TOAs to exclude from transient
            search.
        tau_rec_days (float): Recovery timescale for the exponential decay
            (days).
        window_mult (float): Window length multiplier relative to
            ``tau_rec_days``.
        min_points (int): Minimum number of points required in a candidate
            window.
        delta_chi2_thresh (float): Minimum Δχ² to accept a candidate.
        suppress_overlap (bool): If True, suppress overlapping transient
            assignments.

    Returns:
        pandas.DataFrame: Copy with columns ``transient_id``,
        ``transient_amp``, ``transient_t0``, and ``transient_delta_chi2``
        added.

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
        tt = t[in_win] - t0
        f = np.exp(-tt / tau_rec_days)
        model = A * f
        z_pt = np.full_like(t, np.nan, dtype=float)
        sig = s[in_win]
        good = np.isfinite(sig) & (sig > 0)
        z_pt[in_win] = np.nan
        if np.any(good):
            z_pt[in_win] = np.where(good, np.abs(model) / sig, np.nan)
        member = in_win.copy()
        if np.isfinite(member_eta):
            member &= (z_pt >= float(member_eta))
        d.loc[member, "transient_id"] = k
        d.loc[member, "transient_amp"] = A
        d.loc[member, "transient_t0"] = t0
        d.loc[member, "transient_delta_chi2"] = delta

        if instrument:
            zf = z_pt[np.isfinite(z_pt)]
            if len(zf):
                info_str = (
                    f"transient_id={k} t0={t0:.6f} A={A:.3g} "
                    f"n_assign={int(np.count_nonzero(member))} "
                    f"z_pt[min/med/max]={np.nanmin(zf):.3g}/{np.nanmedian(zf):.3g}/{np.nanmax(zf):.3g} "
                    f"frac<1={float(np.mean(zf < 1.0)):.3g} frac<2={float(np.mean(zf < 2.0)):.3g}"
                )
                try:
                    from pqc.utils.logging import info
                    info(info_str)
                    if np.mean(zf < 1.0) > 0.5:
                        from pqc.utils.logging import warn
                        warn("Transient membership has >50% members with z_pt<1.0; check membership criteria.")
                except Exception:
                    pass

    return d
