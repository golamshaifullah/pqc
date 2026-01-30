import numpy as np
import pandas as pd

from pqc.detect.feature_structure import detect_binned_structure, detrend_residuals_binned


def test_detect_binned_structure_zero_signal():
    df = pd.DataFrame(
        {
            "feature": np.linspace(0.0, 1.0, 48),
            "resid": np.zeros(48),
            "sigma": np.ones(48),
        }
    )
    res = detect_binned_structure(df, "feature", nbins=6)
    assert res["dof"] > 0
    assert np.isfinite(res["chi2"])
    assert res["chi2"] < 1e-6


def test_detrend_residuals_binned_removes_bin_means():
    x = np.concatenate([np.zeros(10), np.ones(10)])
    y = np.concatenate([np.ones(10) * 2.0, np.ones(10) * -1.0])
    df = pd.DataFrame({"feature": x, "resid": y, "sigma": np.ones_like(y)})
    out = detrend_residuals_binned(df, "feature", nbins=2, min_per_bin=3)
    means = out.groupby("feature")["resid_detrended"].mean().to_numpy()
    assert np.all(np.abs(means) < 1e-6)
