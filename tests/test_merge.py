import pandas as pd

from pqc.io.merge import merge_time_and_meta


def test_merge_plain_mjd_fallback_preserves_unmatched_row_identity():
    df_time = pd.DataFrame(
        {
            "mjd": [1.0, 2.0, 3.0],
            "freq": [1400.0, 1500.0, 1600.0],
            "resid": [0.1, 0.2, 0.3],
            "sigma": [1.0, 1.0, 1.0],
        }
    )
    df_meta = pd.DataFrame(
        {
            "mjd": [1.0, 2.0, 3.0],
            "sat_corr": [1.0, 200.0, 3.0],
            "freq": [1400.0, 1500.0, 1600.0],
            "filename": ["a.tim", "b.tim", "c.tim"],
            "_timfile": ["a.tim", "b.tim", "c.tim"],
        }
    )

    out = merge_time_and_meta(df_time, df_meta, tol_days=1e-6, freq_tol_mhz=1.0)

    assert out["filename"].tolist() == ["a.tim", "b.tim", "c.tim"]
    assert out["_timfile"].tolist() == ["a.tim", "b.tim", "c.tim"]
