import numpy as np
import pandas as pd

from pqc.config import (
    BadMeasConfig,
    OutlierGateConfig,
    PreprocConfig,
    RobustOutlierConfig,
    StepConfig,
    StructureConfig,
    TransientConfig,
)
from pqc.pipeline import _run_detection_stage


def test_dm_dvt_flags_with_clamped_k():
    n = 12
    mjd = 59000.0 + np.arange(n, dtype=float)
    sigma = np.ones(n, dtype=float)
    resid = np.zeros(n, dtype=float)
    resid[3] = 5.0

    df = pd.DataFrame(
        {
            "mjd": mjd,
            "day": np.floor(mjd).astype(int),
            "resid": resid,
            "sigma": sigma,
            "freq": np.full(n, 1400.0),
            "group": ["G"] * n,
        }
    )

    out = _run_detection_stage(
        df,
        backend_col="group",
        bad_cfg=BadMeasConfig(tau_corr_days=0.0, fdr_q=0.0, mark_only_worst_per_day=True),
        tr_cfg=TransientConfig(min_points=10_000, delta_chi2_thresh=1e9),
        struct_cfg=StructureConfig(mode="none"),
        step_cfg=StepConfig(enabled=False),
        dm_cfg=StepConfig(enabled=False),
        robust_cfg=RobustOutlierConfig(enabled=False),
        preproc_cfg=PreprocConfig(),
        gate_cfg=OutlierGateConfig(enabled=False),
    )

    assert "bad_dm_dvt" in out.columns
    assert "bad_dm_dvt_label" in out.columns
    assert np.isclose(float(out["bad_dm_dvt_k"].iloc[0]), 1.0)
    assert bool(out.loc[3, "bad_dm_dvt"]) is True
    assert out.loc[3, "bad_dm_dvt_label"] == "BAD_DM_DVT+20"
