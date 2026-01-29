import numpy as np
import pandas as pd

from pqc.detect.step_changes import detect_step


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
