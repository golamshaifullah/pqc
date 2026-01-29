import numpy as np
import pandas as pd

from pqc.config import BadMeasConfig, TransientConfig, StructureConfig, StepConfig, RobustOutlierConfig, PreprocConfig, OutlierGateConfig
from pqc.pipeline import _run_detection_stage


def _make_event_df():
    n = 20
    mjd = 59000.0 + np.arange(n, dtype=float)
    day = np.floor(mjd).astype(int)
    sigma = np.ones(n, dtype=float)
    freq = np.full(n, 100.0)

    # Transient
    t0_tr = mjd[2]
    A = 5.0
    tau = 2.0
    transient = np.where(mjd >= t0_tr, A * np.exp(-(mjd - t0_tr) / tau), 0.0)

    # Step
    t0_step = mjd[10]
    step = np.where(mjd >= t0_step, 3.0, 0.0)

    # DM step (amp in y-space -> residual = amp / freq^2)
    t0_dm = mjd[14]
    dm_amp = 40000.0
    dm = np.where(mjd >= t0_dm, dm_amp / (freq ** 2), 0.0)

    resid = transient + step + dm

    bad_idx = 1
    resid[bad_idx] += 30.0

    df = pd.DataFrame(
        {
            "mjd": mjd,
            "day": day,
            "resid": resid,
            "sigma": sigma,
            "freq": freq,
            "group": np.array(["G1"] * n),
            "orbital_phase": (mjd / 30.0) % 1.0,
            "solar_elongation_deg": np.linspace(30.0, 150.0, n),
        }
    )
    df["true_bad"] = False
    df.loc[bad_idx, "true_bad"] = True
    return df, bad_idx


def test_event_membership_not_bad_point():
    df, bad_idx = _make_event_df()

    bad_cfg = BadMeasConfig(tau_corr_days=0.0, fdr_q=0.0, mark_only_worst_per_day=True)
    tr_cfg = TransientConfig(tau_rec_days=2.0, window_mult=5.0, min_points=4, delta_chi2_thresh=0.1, member_eta=2.0)
    struct_cfg = StructureConfig(mode="none")
    step_cfg = StepConfig(enabled=True, min_points=3, delta_chi2_thresh=0.1, member_eta=2.0, member_tmax_days=5.0)
    dm_cfg = StepConfig(enabled=True, min_points=3, delta_chi2_thresh=0.1, member_eta=2.0, member_tmax_days=5.0)
    robust_cfg = RobustOutlierConfig(enabled=True, z_thresh=6.0, scope="backend")

    out = _run_detection_stage(
        df,
        backend_col="group",
        bad_cfg=bad_cfg,
        tr_cfg=tr_cfg,
        struct_cfg=struct_cfg,
        step_cfg=step_cfg,
        dm_cfg=dm_cfg,
        robust_cfg=robust_cfg,
        preproc_cfg=PreprocConfig(),
        gate_cfg=OutlierGateConfig(),
    )

    assert out["event_member"].any()
    assert int(out["bad_point"].sum()) == 1
    assert bool(out.loc[bad_idx, "bad_point"]) is True
    assert not bool((out["event_member"] & out["bad_point"]).any())


def test_global_step_membership_group_specific():
    n = 24
    mjd = 59000.0 + np.arange(n, dtype=float)
    day = np.floor(mjd).astype(int)
    group = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    mjd = np.concatenate([mjd[: n // 2], mjd[: n // 2]])
    day = np.concatenate([day[: n // 2], day[: n // 2]])

    resid = np.zeros(n, dtype=float)
    resid[mjd >= mjd[8]] = 1.0
    sigma = np.ones(n, dtype=float)
    sigma[group == "A"] = 0.1
    sigma[group == "B"] = 10.0

    df = pd.DataFrame(
        {
            "mjd": mjd,
            "day": day,
            "resid": resid,
            "sigma": sigma,
            "freq": np.full(n, 100.0),
            "group": group,
        }
    )

    out = _run_detection_stage(
        df,
        backend_col="group",
        bad_cfg=BadMeasConfig(tau_corr_days=0.0, fdr_q=0.0, mark_only_worst_per_day=True),
        tr_cfg=TransientConfig(tau_rec_days=2.0, window_mult=5.0, min_points=4, delta_chi2_thresh=0.1, member_eta=2.0),
        struct_cfg=StructureConfig(mode="none"),
        step_cfg=StepConfig(enabled=True, min_points=3, delta_chi2_thresh=0.1, member_eta=2.0, member_tmax_days=5.0, scope="global"),
        dm_cfg=StepConfig(enabled=False),
        robust_cfg=RobustOutlierConfig(enabled=False),
        preproc_cfg=PreprocConfig(),
        gate_cfg=OutlierGateConfig(),
    )

    member_a = (out["group"] == "A") & (out["step_informative"].fillna(False))
    member_b = (out["group"] == "B") & (out["step_informative"].fillna(False))
    assert member_a.any()
    assert not member_b.any()


def test_global_transient_membership_group_specific():
    n = 24
    mjd = 59000.0 + np.arange(n, dtype=float)
    day = np.floor(mjd).astype(int)
    group = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    mjd = np.concatenate([mjd[: n // 2], mjd[: n // 2]])
    day = np.concatenate([day[: n // 2], day[: n // 2]])

    resid = np.zeros(n, dtype=float)
    t0 = mjd[8]
    tau = 3.0
    resid[mjd >= t0] = 5.0 * np.exp(-(mjd[mjd >= t0] - t0) / tau)
    sigma = np.ones(n, dtype=float)
    sigma[group == "A"] = 0.1
    sigma[group == "B"] = 10.0

    df = pd.DataFrame(
        {
            "mjd": mjd,
            "day": day,
            "resid": resid,
            "sigma": sigma,
            "freq": np.full(n, 100.0),
            "group": group,
        }
    )

    out = _run_detection_stage(
        df,
        backend_col="group",
        bad_cfg=BadMeasConfig(tau_corr_days=0.0, fdr_q=0.0, mark_only_worst_per_day=True),
        tr_cfg=TransientConfig(tau_rec_days=tau, window_mult=5.0, min_points=4, delta_chi2_thresh=0.1, member_eta=2.0, scope="global"),
        struct_cfg=StructureConfig(mode="none"),
        step_cfg=StepConfig(enabled=False),
        dm_cfg=StepConfig(enabled=False),
        robust_cfg=RobustOutlierConfig(enabled=False),
        preproc_cfg=PreprocConfig(),
        gate_cfg=OutlierGateConfig(),
    )

    member_a = (out["group"] == "A") & (out["transient_global_id"] >= 0)
    member_b = (out["group"] == "B") & (out["transient_global_id"] >= 0)
    assert member_a.any()
    assert not member_b.any()


def test_global_step_applicable_not_informative():
    n = 30
    mjd = 59000.0 + np.arange(n, dtype=float)
    day = np.floor(mjd).astype(int)
    group = np.array(["A"] * n)
    t0_idx = 12
    resid = np.zeros(n, dtype=float)
    resid[t0_idx:] = 0.5
    sigma = np.ones(n, dtype=float)
    sigma[:5] = 0.1

    df = pd.DataFrame(
        {
            "mjd": mjd,
            "day": day,
            "resid": resid,
            "sigma": sigma,
            "freq": np.full(n, 100.0),
            "group": group,
        }
    )

    out = _run_detection_stage(
        df,
        backend_col="group",
        bad_cfg=BadMeasConfig(tau_corr_days=0.0, fdr_q=0.0, mark_only_worst_per_day=True),
        tr_cfg=TransientConfig(tau_rec_days=2.0, window_mult=5.0, min_points=4, delta_chi2_thresh=0.1, member_eta=2.0),
        struct_cfg=StructureConfig(mode="none"),
        step_cfg=StepConfig(enabled=True, min_points=3, delta_chi2_thresh=0.1, member_eta=1.0, member_tmax_days=10.0, scope="global"),
        dm_cfg=StepConfig(enabled=False),
        robust_cfg=RobustOutlierConfig(enabled=False),
        preproc_cfg=PreprocConfig(),
        gate_cfg=OutlierGateConfig(),
    )

    applicable = out["step_applicable"].fillna(False).to_numpy()
    informative = out["step_informative"].fillna(False).to_numpy()
    assert np.count_nonzero(applicable) > np.count_nonzero(informative)
    assert np.any(applicable & ~informative)
