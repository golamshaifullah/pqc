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
from pqc.detect.bad_measurements import detect_bad
from pqc.detect.transients import scan_transients
from pqc.pipeline import _run_detection_stage


def _make_synth_df(seed: int = 123) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    n = 240
    dt = rng.uniform(0.2, 2.0, size=n)
    mjd = 59000.0 + np.cumsum(dt)
    day = np.floor(mjd).astype(int)

    phase = (mjd / 30.0) % 1.0
    solar = 30.0 + 150.0 * (0.5 * (1.0 + np.sin(2.0 * np.pi * phase + 0.3)))

    sigma = 1.0 * (1.0 + 0.3 * (solar / 180.0))
    sigma_true = sigma * (1.0 + 2.5 * (solar / 180.0))

    tau = 1.2
    state = np.zeros(n)
    for i in range(n):
        if i == 0:
            state[i] = rng.normal(scale=sigma_true[i])
        else:
            dt_i = mjd[i] - mjd[i - 1]
            phi = np.exp(-dt_i / tau)
            innov = rng.normal(scale=sigma_true[i])
            state[i] = phi * state[i - 1] + np.sqrt(max(0.0, 1.0 - phi * phi)) * innov

    mean = 6.0 * np.sin(2.0 * np.pi * phase)
    resid = mean + state

    t0 = mjd[int(0.6 * n)]
    tau_rec = 5.0
    resid += np.where(mjd >= t0, 6.0 * np.exp(-(mjd - t0) / tau_rec), 0.0)

    bad_idx = int(0.3 * n)
    resid[bad_idx] += 12.0

    freq = 1400.0 + 100.0 * np.cos(2.0 * np.pi * phase)

    df = pd.DataFrame(
        {
            "mjd": mjd,
            "day": day,
            "resid": resid,
            "sigma": sigma,
            "group": np.array(["G1"] * n),
            "orbital_phase": phase,
            "solar_elongation_deg": solar,
            "freq": freq,
        }
    )
    df["true_bad"] = False
    df.loc[bad_idx, "true_bad"] = True
    return df, bad_idx


def _make_deterministic_df() -> tuple[pd.DataFrame, int]:
    n = 12
    mjd = 59000.0 + np.arange(n, dtype=float)
    day = np.floor(mjd).astype(int)

    orbital_phase = np.array(
        [0.05, 0.10, 0.12, 0.30, 0.32, 0.33, 0.34, 0.60, 0.62, 0.65, 0.80, 0.85]
    )
    sigma = np.ones(n, dtype=float)
    resid = np.array(
        [0.50, 0.60, 0.40, 10.50, 10.60, 10.40, 10.55, 0.45, 0.52, 0.48, 30.50, 0.58], dtype=float
    )
    bad_idx = 10

    df = pd.DataFrame(
        {
            "mjd": mjd,
            "day": day,
            "resid": resid,
            "sigma": sigma,
            "group": np.array(["G1"] * n),
            "orbital_phase": orbital_phase,
            "solar_elongation_deg": np.linspace(30.0, 150.0, n),
            "freq": np.full(n, 1400.0),
        }
    )
    df["true_bad"] = False
    df.loc[bad_idx, "true_bad"] = True
    return df, bad_idx


def test_preproc_reduces_false_ou_flags_and_keeps_true_bad():
    df, bad_idx = _make_deterministic_df()

    bad_cfg = BadMeasConfig(tau_corr_days=0.0, fdr_q=0.2, mark_only_worst_per_day=True)
    tr_cfg = TransientConfig(tau_rec_days=5.0, delta_chi2_thresh=12.0, min_points=6)
    struct_cfg = StructureConfig(mode="none")
    step_cfg = StepConfig(enabled=False)
    dm_cfg = StepConfig(enabled=False)
    robust_cfg = RobustOutlierConfig(enabled=False)

    raw = _run_detection_stage(
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

    preproc_cfg = PreprocConfig(
        detrend_features=("orbital_phase",),
        rescale_feature=None,
        condition_on=("group",),
        use_preproc_for=("ou",),
        nbins=12,
        min_per_bin=3,
    )
    proc = _run_detection_stage(
        df,
        backend_col="group",
        bad_cfg=bad_cfg,
        tr_cfg=tr_cfg,
        struct_cfg=struct_cfg,
        step_cfg=step_cfg,
        dm_cfg=dm_cfg,
        robust_cfg=robust_cfg,
        preproc_cfg=preproc_cfg,
        gate_cfg=OutlierGateConfig(),
    )

    assert len(proc) == len(df)
    assert (proc["group"].to_numpy() == df["group"].to_numpy()).all()
    raw_bad = int(raw["bad_ou"].sum())
    proc_bad = int(proc["bad_ou"].sum())
    assert raw_bad == 5
    assert proc_bad == 1
    assert bool(proc.loc[bad_idx, "bad_ou"]) is True


def test_structure_present_does_not_change_bad_flags():
    df, _ = _make_synth_df()
    bad_cfg = BadMeasConfig(tau_corr_days=0.3, fdr_q=0.2, mark_only_worst_per_day=True)
    tr_cfg = TransientConfig(tau_rec_days=5.0, delta_chi2_thresh=12.0, min_points=6)
    step_cfg = StepConfig(enabled=False)
    dm_cfg = StepConfig(enabled=False)
    robust_cfg = RobustOutlierConfig(enabled=False)

    none_struct = _run_detection_stage(
        df,
        backend_col="group",
        bad_cfg=bad_cfg,
        tr_cfg=tr_cfg,
        struct_cfg=StructureConfig(mode="none"),
        step_cfg=step_cfg,
        dm_cfg=dm_cfg,
        robust_cfg=robust_cfg,
        preproc_cfg=PreprocConfig(),
        gate_cfg=OutlierGateConfig(),
    )
    test_struct = _run_detection_stage(
        df,
        backend_col="group",
        bad_cfg=bad_cfg,
        tr_cfg=tr_cfg,
        struct_cfg=StructureConfig(
            mode="test", structure_features=("orbital_phase",), p_thresh=0.05
        ),
        step_cfg=step_cfg,
        dm_cfg=dm_cfg,
        robust_cfg=robust_cfg,
        preproc_cfg=PreprocConfig(),
        gate_cfg=OutlierGateConfig(),
    )

    assert np.array_equal(none_struct["bad_ou"].to_numpy(), test_struct["bad_ou"].to_numpy())
    assert test_struct["structure_present_orbital_phase"].any()


def test_defaults_match_raw_detectors_when_preproc_disabled():
    df, _ = _make_synth_df()
    bad_cfg = BadMeasConfig(tau_corr_days=0.3, fdr_q=0.2, mark_only_worst_per_day=True)
    tr_cfg = TransientConfig(tau_rec_days=5.0, delta_chi2_thresh=12.0, min_points=6)
    struct_cfg = StructureConfig(mode="none")
    step_cfg = StepConfig(enabled=False)
    dm_cfg = StepConfig(enabled=False)
    robust_cfg = RobustOutlierConfig(enabled=False)

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

    manual = []
    for _, sub in df.groupby("group"):
        sub1 = detect_bad(
            sub,
            tau_corr_days=bad_cfg.tau_corr_days,
            fdr_q=bad_cfg.fdr_q,
            mark_only_worst_per_day=bad_cfg.mark_only_worst_per_day,
            resid_col="resid",
            sigma_col="sigma",
        )
        sub2 = scan_transients(
            sub1,
            tau_rec_days=tr_cfg.tau_rec_days,
            window_mult=tr_cfg.window_mult,
            min_points=tr_cfg.min_points,
            delta_chi2_thresh=tr_cfg.delta_chi2_thresh,
            suppress_overlap=tr_cfg.suppress_overlap,
            resid_col="resid",
            sigma_col="sigma",
            exclude_bad_col="bad",
        )
        manual.append(sub2)
    manual_df = pd.concat(manual, axis=0).sort_values("mjd").reset_index(drop=True)

    assert np.array_equal(out["bad"].to_numpy(), manual_df["bad"].to_numpy())
    assert np.array_equal(out["bad_day"].to_numpy(), manual_df["bad_day"].to_numpy())
    assert np.array_equal(out["transient_id"].to_numpy(), manual_df["transient_id"].to_numpy())


def test_detector_provenance_columns_change_with_preproc():
    df, _ = _make_synth_df()
    bad_cfg = BadMeasConfig(tau_corr_days=0.3, fdr_q=0.2, mark_only_worst_per_day=True)
    tr_cfg = TransientConfig(tau_rec_days=5.0, delta_chi2_thresh=12.0, min_points=6)
    struct_cfg = StructureConfig(mode="none")
    step_cfg = StepConfig(enabled=False)
    dm_cfg = StepConfig(enabled=False)
    robust_cfg = RobustOutlierConfig(enabled=False)

    preproc_cfg = PreprocConfig(
        detrend_features=("orbital_phase",),
        rescale_feature=None,
        condition_on=("group",),
        use_preproc_for=("ou", "transient", "mad"),
        nbins=12,
        min_per_bin=3,
    )

    out = _run_detection_stage(
        df,
        backend_col="group",
        bad_cfg=bad_cfg,
        tr_cfg=tr_cfg,
        struct_cfg=struct_cfg,
        step_cfg=step_cfg,
        dm_cfg=dm_cfg,
        robust_cfg=robust_cfg,
        preproc_cfg=preproc_cfg,
        gate_cfg=OutlierGateConfig(),
    )

    assert out["ou_used_resid_col"].iloc[0] == "resid_detr"
    assert out["transient_used_resid_col"].iloc[0] == "resid_detr"
    assert out["mad_used_resid_col"].iloc[0] == "resid_detr"


def test_transient_detected_with_preproc():
    df, _ = _make_synth_df()
    bad_cfg = BadMeasConfig(tau_corr_days=0.3, fdr_q=0.2, mark_only_worst_per_day=True)
    tr_cfg = TransientConfig(tau_rec_days=5.0, delta_chi2_thresh=5.0, min_points=6)
    struct_cfg = StructureConfig(mode="none")
    step_cfg = StepConfig(enabled=False)
    dm_cfg = StepConfig(enabled=False)
    robust_cfg = RobustOutlierConfig(enabled=False)

    preproc_cfg = PreprocConfig(
        detrend_features=("orbital_phase",),
        rescale_feature="solar_elongation_deg",
        condition_on=("group",),
        use_preproc_for=("transient",),
        nbins=12,
        min_per_bin=5,
    )

    out = _run_detection_stage(
        df,
        backend_col="group",
        bad_cfg=bad_cfg,
        tr_cfg=tr_cfg,
        struct_cfg=struct_cfg,
        step_cfg=step_cfg,
        dm_cfg=dm_cfg,
        robust_cfg=robust_cfg,
        preproc_cfg=preproc_cfg,
        gate_cfg=OutlierGateConfig(),
    )

    assert (out["transient_id"] >= 0).any()
