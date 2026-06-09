import numpy as np
import pandas as pd

from pqc.detect.step_changes import detect_dm_step, detect_step


def test_step_membership_respects_window():
    n = 20
    mjd = 59000.0 + np.arange(n, dtype=float)
    sigma = np.ones(n, dtype=float)
    resid = np.zeros(n, dtype=float)
    t0_idx = 10
    resid[t0_idx:] = 5.0

    df = pd.DataFrame({"mjd": mjd, "resid": resid, "sigma": sigma})

    out = detect_step(
        df,
        mjd_col="mjd",
        resid_col="resid",
        sigma_col="sigma",
        min_points=3,
        delta_chi2_thresh=0.1,
        member_eta=1.0,
        member_tmax_days=2.0,
        prefix="step",
    )

    applicable = out["step_applicable"].to_numpy()
    informative = out["step_informative"].to_numpy()
    # Only within [t0, t0+2] should be members (inclusive)
    t0 = out["step_t0"].iloc[0]
    expected = (mjd >= t0) & (mjd <= t0 + 2.0)
    assert np.array_equal(applicable, expected)
    assert np.array_equal(informative, expected)


def test_step_min_span_rejects_short_candidate_side():
    mjd = 59000.0 + np.concatenate(
        [np.arange(10, dtype=float), 10.0 + 0.1 * np.arange(10, dtype=float)]
    )
    sigma = np.ones_like(mjd)
    resid = np.zeros_like(mjd)
    resid[mjd >= 59010.0] = 5.0
    df = pd.DataFrame({"mjd": mjd, "resid": resid, "sigma": sigma})

    detected = detect_step(
        df,
        min_points=3,
        min_span_days=None,
        delta_chi2_thresh=0.1,
        member_eta=1.0,
        member_tmax_days=0.1,
    )
    rejected = detect_step(
        df,
        min_points=3,
        min_span_days=2.0,
        delta_chi2_thresh=0.1,
        member_eta=1.0,
        member_tmax_days=0.1,
    )

    assert np.isfinite(detected["step_t0"].iloc[0])
    assert not np.isfinite(rejected["step_t0"].iloc[0])
    assert not rejected["step_applicable"].any()


def test_dm_step_min_span_rejects_short_candidate_side():
    mjd = 59000.0 + np.concatenate(
        [np.arange(10, dtype=float), 10.0 + 0.1 * np.arange(10, dtype=float)]
    )
    sigma = np.ones_like(mjd)
    freq = np.full_like(mjd, 100.0)
    resid = np.zeros_like(mjd)
    resid[mjd >= 59010.0] = 40000.0 / (freq[mjd >= 59010.0] ** 2)
    df = pd.DataFrame({"mjd": mjd, "resid": resid, "sigma": sigma, "freq": freq})

    detected = detect_dm_step(
        df,
        min_points=3,
        min_span_days=None,
        delta_chi2_thresh=0.1,
        member_eta=0.1,
        member_tmax_days=0.1,
    )
    rejected = detect_dm_step(
        df,
        min_points=3,
        min_span_days=2.0,
        delta_chi2_thresh=0.1,
        member_eta=0.1,
        member_tmax_days=0.1,
    )

    assert np.isfinite(detected["dm_step_t0"].iloc[0])
    assert not np.isfinite(rejected["dm_step_t0"].iloc[0])
    assert not rejected["dm_step_applicable"].any()
