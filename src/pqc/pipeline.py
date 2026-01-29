"""Run the end-to-end PQC pipeline.

The primary entry point is :func:`run_pipeline`, which performs a standard PTA
QC workflow:

1. Parse timfiles (including INCLUDE recursion).
2. Load libstempo arrays (TOA/residual/error/frequency).
3. Merge arrays with timfile metadata by nearest MJD.
4. Ensure backend keys (``sys``/``group``) exist.
5. Add feature columns (orbital phase, solar elongation, optional alt/airmass/
   parallactic angle, optional frequency bins).
6. Detect bad measurements and transient events per backend group.

See Also:
    pqc.io.timfile.parse_all_timfiles: Parse timfile metadata.
    pqc.io.libstempo_loader.load_libstempo: Load timing arrays via libstempo.
    pqc.io.merge.merge_time_and_meta: Merge timing arrays with metadata.
    pqc.features.backend_keys.ensure_sys_group: Normalize backend keys.
    pqc.features.feature_extraction.add_feature_columns: Feature extraction.
    pqc.detect.bad_measurements.detect_bad: Bad measurement detection.
    pqc.detect.feature_structure: Feature-domain structure tests/detrending.
    pqc.detect.transients.scan_transients: Transient detection.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from pqc.config import (
    BadMeasConfig,
    FeatureConfig,
    MergeConfig,
    StructureConfig,
    TransientConfig,
    StepConfig,
    RobustOutlierConfig,
    PreprocConfig,
    OutlierGateConfig,
)
from pqc.io.timfile import parse_all_timfiles
from pqc.io.libstempo_loader import load_libstempo
from pqc.io.merge import merge_time_and_meta
from pqc.features.backend_keys import ensure_sys_group
from pqc.features.feature_extraction import add_feature_columns
from pqc.detect.bad_measurements import detect_bad
from pqc.detect.feature_structure import detect_binned_structure, detrend_residuals_binned
from pqc.detect.transients import scan_transients
from pqc.detect.step_changes import detect_step, detect_dm_step
from pqc.detect.robust_outliers import detect_robust_outliers
from pqc.preproc.mean_model import detrend_by_features
from pqc.preproc.variance_model import rescale_by_feature
from pqc.utils.logging import info, warn

_DETECTOR_NAMES = {"ou", "transient", "mad", "step", "dmstep"}


def _parse_use_preproc(use_preproc_for: tuple[str, ...] | list[str]) -> set[str]:
    if not use_preproc_for:
        return set()
    return {u.strip().lower() for u in use_preproc_for if u.strip()}


def _choose_resid_sigma(
    detector: str,
    *,
    use_preproc: set[str],
    base_resid_col: str,
    base_sigma_col: str,
    proc_resid_col: str | None,
    proc_sigma_col: str | None,
) -> tuple[str, str]:
    if detector in use_preproc and proc_resid_col is not None:
        return proc_resid_col, proc_sigma_col or base_sigma_col
    return base_resid_col, base_sigma_col


def _run_detection_stage(
    df: pd.DataFrame,
    *,
    backend_col: str,
    bad_cfg: BadMeasConfig,
    tr_cfg: TransientConfig,
    struct_cfg: StructureConfig,
    step_cfg: StepConfig,
    dm_cfg: StepConfig,
    robust_cfg: RobustOutlierConfig,
    preproc_cfg: PreprocConfig,
    gate_cfg: OutlierGateConfig,
) -> pd.DataFrame:
    out = []
    do_detrend = struct_cfg.mode in ("detrend", "both")
    do_test = struct_cfg.mode in ("test", "both")
    circ = set(struct_cfg.circular_features)
    structure_cols = list(struct_cfg.structure_group_cols or (backend_col,))
    structure_cols = [c for c in structure_cols if c in df.columns]
    if not structure_cols:
        structure_cols = [backend_col]

    df_work = df.copy()
    resid_base_col = "resid"

    if do_detrend:
        df_work["resid_detrended"] = df_work["resid"].to_numpy(dtype=float)
        resid_base_col = "resid_detrended"
        for _, sub in df_work.groupby(structure_cols):
            idx = sub.index
            for feat in struct_cfg.detrend_features:
                sub = detrend_residuals_binned(
                    sub,
                    feat,
                    resid_col=resid_base_col,
                    sigma_col="sigma",
                    nbins=struct_cfg.nbins,
                    circular=feat in circ,
                    min_per_bin=struct_cfg.min_per_bin,
                    out_col=resid_base_col,
                )
            df_work.loc[idx, resid_base_col] = sub[resid_base_col].to_numpy()

    if do_test:
        for _, sub in df_work.groupby(structure_cols):
            idx = sub.index
            for feat in struct_cfg.structure_features:
                base = f"structure_{feat}"
                present_col = f"structure_present_{feat}"
                if feat not in sub.columns:
                    df_work.loc[idx, f"{base}_chi2"] = np.nan
                    df_work.loc[idx, f"{base}_dof"] = 0
                    df_work.loc[idx, f"{base}_p"] = np.nan
                    df_work.loc[idx, f"{base}_present"] = False
                    df_work.loc[idx, present_col] = False
                    continue
                res = detect_binned_structure(
                    sub,
                    feat,
                    resid_col=resid_base_col,
                    sigma_col="sigma",
                    nbins=struct_cfg.nbins,
                    circular=feat in circ,
                    min_per_bin=struct_cfg.min_per_bin,
                )
                p_like = res.get("p_like", np.nan)
                present = bool(np.isfinite(p_like) and p_like < struct_cfg.p_thresh)
                df_work.loc[idx, f"{base}_chi2"] = res.get("chi2", np.nan)
                df_work.loc[idx, f"{base}_dof"] = res.get("dof", 0)
                df_work.loc[idx, f"{base}_p"] = p_like
                df_work.loc[idx, f"{base}_present"] = present
                df_work.loc[idx, present_col] = present

    df_work["preproc_tag"] = ""
    df_work["preproc_notes"] = ""

    proc_resid_col = None
    proc_sigma_col = None

    preproc_feats = tuple(preproc_cfg.detrend_features)
    circular_map = {f: (f in preproc_cfg.circular_features) for f in preproc_feats}
    group_cols = tuple(c for c in preproc_cfg.condition_on if c in df_work.columns)
    if preproc_feats:
        df_work, _ = detrend_by_features(
            df_work,
            preproc_feats,
            group_cols=group_cols or (backend_col,),
            resid_col=resid_base_col,
            sigma_col="sigma",
            nbins_map={f: preproc_cfg.nbins for f in preproc_feats},
            circular_map=circular_map,
            min_per_bin=preproc_cfg.min_per_bin,
            out_col="resid_detr",
            store_models=True,
        )
        proc_resid_col = "resid_detr"
        proc_sigma_col = "sigma"
        df_work["preproc_tag"] = f"detrend:{','.join(preproc_feats)}"
        df_work["preproc_notes"] = f"group_cols={group_cols or (backend_col,)}; nbins={preproc_cfg.nbins}; min_per_bin={preproc_cfg.min_per_bin}"

    if preproc_cfg.rescale_feature:
        rescale_feat = preproc_cfg.rescale_feature
        base_for_scale = proc_resid_col or resid_base_col
        df_work = rescale_by_feature(
            df_work,
            rescale_feat,
            group_cols=group_cols or (backend_col,),
            resid_col=base_for_scale,
            sigma_col="sigma",
            nbins=preproc_cfg.nbins,
            circular=rescale_feat in preproc_cfg.circular_features,
            min_per_bin=preproc_cfg.min_per_bin,
            out_resid_col="resid_proc",
            out_sigma_col="sigma_proc",
        )
        proc_resid_col = "resid_proc"
        proc_sigma_col = "sigma_proc"
        tag = df_work["preproc_tag"].iloc[0] if len(df_work) else ""
        sep = "|" if tag else ""
        df_work["preproc_tag"] = f"{tag}{sep}rescale:{rescale_feat}"
        df_work["preproc_notes"] = (
            f"group_cols={group_cols or (backend_col,)}; nbins={preproc_cfg.nbins}; min_per_bin={preproc_cfg.min_per_bin}"
        )

    use_preproc = _parse_use_preproc(preproc_cfg.use_preproc_for)
    use_preproc = use_preproc & _DETECTOR_NAMES

    ou_resid, ou_sigma = _choose_resid_sigma(
        "ou",
        use_preproc=use_preproc,
        base_resid_col=resid_base_col,
        base_sigma_col="sigma",
        proc_resid_col=proc_resid_col,
        proc_sigma_col=proc_sigma_col,
    )
    tr_resid, tr_sigma = _choose_resid_sigma(
        "transient",
        use_preproc=use_preproc,
        base_resid_col=resid_base_col,
        base_sigma_col="sigma",
        proc_resid_col=proc_resid_col,
        proc_sigma_col=proc_sigma_col,
    )
    step_resid, step_sigma = _choose_resid_sigma(
        "step",
        use_preproc=use_preproc,
        base_resid_col=resid_base_col,
        base_sigma_col="sigma",
        proc_resid_col=proc_resid_col,
        proc_sigma_col=proc_sigma_col,
    )
    dm_resid, dm_sigma = _choose_resid_sigma(
        "dmstep",
        use_preproc=use_preproc,
        base_resid_col=resid_base_col,
        base_sigma_col="sigma",
        proc_resid_col=proc_resid_col,
        proc_sigma_col=proc_sigma_col,
    )
    mad_resid, _ = _choose_resid_sigma(
        "mad",
        use_preproc=use_preproc,
        base_resid_col=resid_base_col,
        base_sigma_col="sigma",
        proc_resid_col=proc_resid_col,
        proc_sigma_col=proc_sigma_col,
    )

    df_work["ou_used_resid_col"] = ou_resid
    df_work["ou_used_sigma_col"] = ou_sigma
    df_work["transient_used_resid_col"] = tr_resid
    df_work["transient_used_sigma_col"] = tr_sigma
    df_work["step_used_resid_col"] = step_resid
    df_work["step_used_sigma_col"] = step_sigma
    df_work["dmstep_used_resid_col"] = dm_resid
    df_work["dmstep_used_sigma_col"] = dm_sigma
    df_work["mad_used_resid_col"] = mad_resid

    if step_cfg.enabled and step_cfg.scope in ("global", "both"):
        df_work = detect_step(
            df_work,
            mjd_col="mjd",
            resid_col=step_resid,
            sigma_col=step_sigma,
            min_points=step_cfg.min_points,
            delta_chi2_thresh=step_cfg.delta_chi2_thresh,
            member_eta=step_cfg.member_eta,
            member_tmax_days=step_cfg.member_tmax_days,
            instrument=bool(getattr(step_cfg, "instrument", False)),
            prefix="step_global",
        )
    if dm_cfg.enabled and dm_cfg.scope in ("global", "both"):
        df_work = detect_dm_step(
            df_work,
            mjd_col="mjd",
            resid_col=dm_resid,
            sigma_col=dm_sigma,
            freq_col="freq",
            min_points=dm_cfg.min_points,
            delta_chi2_thresh=dm_cfg.delta_chi2_thresh,
            member_eta=dm_cfg.member_eta,
            member_tmax_days=dm_cfg.member_tmax_days,
            instrument=bool(getattr(dm_cfg, "instrument", False)),
            prefix="dm_step_global",
        )

    if robust_cfg.enabled and robust_cfg.scope in ("global", "both"):
        df_work = detect_robust_outliers(
            df_work,
            resid_col=mad_resid,
            z_thresh=robust_cfg.z_thresh,
            prefix="robust_global",
        )

    for key, sub in df_work.groupby(backend_col):
        sub1 = detect_bad(
            sub,
            tau_corr_days=bad_cfg.tau_corr_days,
            fdr_q=bad_cfg.fdr_q,
            mark_only_worst_per_day=bad_cfg.mark_only_worst_per_day,
            resid_col=ou_resid,
            sigma_col=ou_sigma,
        )
        sub2 = scan_transients(
            sub1,
            tau_rec_days=tr_cfg.tau_rec_days,
            window_mult=tr_cfg.window_mult,
            min_points=tr_cfg.min_points,
            delta_chi2_thresh=tr_cfg.delta_chi2_thresh,
            suppress_overlap=tr_cfg.suppress_overlap,
            resid_col=tr_resid,
            sigma_col=tr_sigma,
            exclude_bad_col="bad",
            member_eta=tr_cfg.member_eta,
            instrument=bool(getattr(tr_cfg, "instrument", False)),
        )
        if step_cfg.enabled and step_cfg.scope in ("backend", "both"):
            sub2 = detect_step(
                sub2,
                mjd_col="mjd",
                resid_col=step_resid,
                sigma_col=step_sigma,
                min_points=step_cfg.min_points,
                delta_chi2_thresh=step_cfg.delta_chi2_thresh,
                member_eta=step_cfg.member_eta,
                member_tmax_days=step_cfg.member_tmax_days,
                instrument=bool(getattr(step_cfg, "instrument", False)),
                prefix="step",
            )
        if dm_cfg.enabled and dm_cfg.scope in ("backend", "both"):
            sub2 = detect_dm_step(
                sub2,
                mjd_col="mjd",
                resid_col=dm_resid,
                sigma_col=dm_sigma,
                freq_col="freq",
                min_points=dm_cfg.min_points,
                delta_chi2_thresh=dm_cfg.delta_chi2_thresh,
                member_eta=dm_cfg.member_eta,
                member_tmax_days=dm_cfg.member_tmax_days,
                instrument=bool(getattr(dm_cfg, "instrument", False)),
                prefix="dm_step",
            )
        if robust_cfg.enabled and robust_cfg.scope in ("backend", "both"):
            sub2 = detect_robust_outliers(
                sub2,
                resid_col=mad_resid,
                z_thresh=robust_cfg.z_thresh,
                prefix="robust",
            )
        out.append(sub2)

    if not out:
        return df.iloc[0:0].copy()

    df_out = pd.concat(out, axis=0).sort_values("mjd").reset_index(drop=True)
    df_out["bad_ou"] = df_out.get("bad", False).fillna(False)
    df_out["bad_mad"] = False
    for col in ("robust_outlier", "robust_global_outlier"):
        if col in df_out.columns:
            df_out["bad_mad"] |= df_out[col].fillna(False)

    if "step_id" not in df_out.columns and "step_global_id" in df_out.columns:
        df_out["step_id"] = df_out["step_global_id"].fillna(-1).astype(int)
    if "dm_step_id" not in df_out.columns and "dm_step_global_id" in df_out.columns:
        df_out["dm_step_id"] = df_out["dm_step_global_id"].fillna(-1).astype(int)

    if "step_id" in df_out.columns:
        df_out["step_id"] = df_out["step_id"].fillna(-1).astype(int)
    else:
        df_out["step_id"] = -1
    if "dm_step_id" in df_out.columns:
        df_out["dm_step_id"] = df_out["dm_step_id"].fillna(-1).astype(int)
    else:
        df_out["dm_step_id"] = -1

    gate_inlier = None
    gate_valid = None
    if gate_cfg.enabled:
        resid_gate_col = gate_cfg.resid_col
        if resid_gate_col is None:
            resid_gate_col = ou_resid
        sigma_gate_col = gate_cfg.sigma_col
        if sigma_gate_col is None:
            sigma_gate_col = ou_sigma
        if resid_gate_col in df_out.columns and sigma_gate_col in df_out.columns:
            r = pd.to_numeric(df_out[resid_gate_col], errors="coerce").to_numpy(dtype=float)
            s = pd.to_numeric(df_out[sigma_gate_col], errors="coerce").to_numpy(dtype=float)
            gate_valid = np.isfinite(r) & np.isfinite(s)
            gate_inlier = gate_valid & (np.abs(r) <= float(gate_cfg.sigma_thresh) * s)
        df_out["outlier_gate_sigma"] = float(gate_cfg.sigma_thresh)
        df_out["outlier_gate_resid_col"] = resid_gate_col
        df_out["outlier_gate_sigma_col"] = sigma_gate_col
        df_out["outlier_gate_inlier"] = False if gate_inlier is None else gate_inlier
        if gate_cfg.resid_col is None and gate_cfg.sigma_col is None:
            if (mad_resid != resid_gate_col):
                warn(f"Outlier gate uses {resid_gate_col}/{sigma_gate_col} but MAD uses {mad_resid}/sigma; set gate columns explicitly to align.")

    df_out["bad_hard"] = False
    if gate_inlier is not None:
        if gate_valid is None:
            gate_valid = np.isfinite(df_out[resid_gate_col]) & np.isfinite(df_out[sigma_gate_col])
        df_out["bad_hard"] = gate_valid & (~gate_inlier)
        if "bad_ou" in df_out.columns:
            df_out.loc[gate_inlier, "bad_ou"] = False
        if "bad_mad" in df_out.columns:
            df_out.loc[gate_inlier, "bad_mad"] = False
        for col in ("robust_outlier", "robust_global_outlier"):
            if col in df_out.columns:
                df_out.loc[gate_inlier, col] = False

    df_out["bad_point"] = False
    df_out["bad_point"] |= df_out.get("bad_ou", False).fillna(False)
    df_out["bad_point"] |= df_out.get("bad_mad", False).fillna(False)
    if "robust_outlier" in df_out.columns:
        df_out["bad_point"] |= df_out["robust_outlier"].fillna(False)
    if "bad_hard" in df_out.columns:
        df_out["bad_point"] |= df_out["bad_hard"].fillna(False)

    df_out["event_member"] = False
    if "transient_id" in df_out.columns:
        df_out["event_member"] |= df_out["transient_id"].fillna(-1).to_numpy() >= 0
    if "step_id" in df_out.columns:
        df_out["event_member"] |= df_out["step_id"].fillna(-1).to_numpy() >= 0
    if "dm_step_id" in df_out.columns:
        df_out["event_member"] |= df_out["dm_step_id"].fillna(-1).to_numpy() >= 0
    df_out["event_member"] &= ~df_out["bad_point"].fillna(False)

    df_out["outlier_any"] = False
    df_out["outlier_any"] |= df_out["bad_point"]
    df_out["outlier_any"] |= df_out["event_member"]

    return df_out


def run_pipeline(
    parfile: str | Path,
    *,
    backend_col: str = "group",
    bad_cfg: BadMeasConfig = BadMeasConfig(),
    tr_cfg: TransientConfig = TransientConfig(),
    merge_cfg: MergeConfig = MergeConfig(),
    feature_cfg: FeatureConfig = FeatureConfig(),
    struct_cfg: StructureConfig = StructureConfig(),
    step_cfg: StepConfig = StepConfig(),
    dm_cfg: StepConfig = StepConfig(),
    robust_cfg: RobustOutlierConfig = RobustOutlierConfig(),
    preproc_cfg: PreprocConfig = PreprocConfig(),
    gate_cfg: OutlierGateConfig = OutlierGateConfig(),
    drop_unmatched: bool = False,
) -> pd.DataFrame:
    """Run the full PTA QC pipeline for a single pulsar.

    This function reads the ``.par`` file and its sibling ``*_all.tim``,
    constructs timing and metadata tables, adds optional feature columns, and
    annotates each TOA with QC flags and transient detections.

    Args:
        parfile (str | Path): Path to the pulsar ``.par`` file. A sibling
            ``*_all.tim`` is required and will be discovered by filename
            convention.
        backend_col (str): Column used to group TOAs for per-backend QC.
        bad_cfg (BadMeasConfig): Configuration for bad-measurement detection.
        tr_cfg (TransientConfig): Configuration for transient detection.
        merge_cfg (MergeConfig): Configuration for timfile/TOA matching.
        feature_cfg (FeatureConfig): Configuration for feature-column extraction.
        struct_cfg (StructureConfig): Configuration for feature-domain structure
            tests/detrending.
        step_cfg (StepConfig): Configuration for step-like offsets in residuals.
        dm_cfg (StepConfig): Configuration for DM-like step offsets (freq-scaled).
        preproc_cfg (PreprocConfig): Configuration for covariate-conditioned
            preprocessing (detrend/rescale) prior to selected detectors.
        gate_cfg (OutlierGateConfig): Configuration for hard sigma gating of
            outlier membership.
        drop_unmatched (bool): If True, drop TOAs whose metadata could not be
            matched.

    Returns:
        pandas.DataFrame: Timing, metadata, and QC annotations. The output
        includes the merged timfile metadata plus ``bad``, ``bad_day``, ``z``,
        ``bad_ou``, ``bad_mad``, ``bad_point``, ``event_member``,
        ``transient_id``, ``step_id``, ``dm_step_id``, and ``outlier_any``
        (deprecated for plotting/summary). If enabled, feature columns such as
        ``orbital_phase`` and ``solar_elongation_deg`` are added, plus
        structure-test summaries (``structure_*`` and
        ``structure_present_<feature>``) and optional preprocessing columns
        (``resid_detrended``, ``resid_detr``, ``resid_proc``, ``sigma_proc``,
        ``preproc_tag``, ``preproc_notes``).

    Raises:
        FileNotFoundError: If ``parfile`` or the matching ``*_all.tim`` is missing.

    Notes:
        This function logs progress via :mod:`pqc.utils.logging`. It does not
        mutate input files, but it will run libstempo, which may be sensitive
        to local tempo2 installation details.

    Examples:
        Run with defaults:

        >>> from pqc.pipeline import run_pipeline
        >>> df = run_pipeline("/data/J1909-3744.par")  # doctest: +SKIP

        Run with custom detection parameters:

        >>> from pqc.config import BadMeasConfig, TransientConfig
        >>> df = run_pipeline(  # doctest: +SKIP
        ...     "/data/J1909-3744.par",
        ...     bad_cfg=BadMeasConfig(tau_corr_days=0.03),
        ...     tr_cfg=TransientConfig(delta_chi2_thresh=40.0),
        ... )
    """
    parfile = Path(parfile)
    if not parfile.exists():
        raise FileNotFoundError(str(parfile))

    all_tim = str(parfile).replace(".par", "_all.tim")
    if not Path(all_tim).exists():
        raise FileNotFoundError(all_tim)

    info("[1/6] Parse timfiles")
    tim_res = parse_all_timfiles(all_tim)
    df_tim = tim_res.df
    if tim_res.dropped_lines:
        warn(f"Dropped {tim_res.dropped_lines} malformed TOA lines while parsing timfiles.")

    info("[2/6] Load libstempo")
    df_time = load_libstempo(parfile)

    info("[3/6] Merge by MJD")
    df = merge_time_and_meta(df_time, df_tim, tol_days=merge_cfg.tol_days)

    n_unmatched = int(df["filename"].isna().sum())
    if n_unmatched:
        warn(
            f"{n_unmatched} TOAs unmatched to tim metadata (within tol={merge_cfg.tol_days} days). "
            + ("Dropping them." if drop_unmatched else "Keeping them with NaN metadata.")
        )
        if drop_unmatched:
            df = df.loc[~df["filename"].isna()].reset_index(drop=True)

    info("[4/6] Ensure sys/group")
    df = ensure_sys_group(df)

    info("[5/6] Add feature columns")
    df = add_feature_columns(
        df,
        parfile,
        mjd_col="mjd",
        add_orb_phase=feature_cfg.add_orbital_phase,
        add_solar=feature_cfg.add_solar_elongation,
        add_elevation=feature_cfg.add_elevation,
        add_airmass=feature_cfg.add_airmass,
        add_parallactic=feature_cfg.add_parallactic_angle,
        add_freq=feature_cfg.add_freq_bin,
        freq_bins=feature_cfg.freq_bins,
        observatory_path=feature_cfg.observatory_path,
    )

    info("[6/6] Detect bad measurements + transients per backend")
    return _run_detection_stage(
        df,
        backend_col=backend_col,
        bad_cfg=bad_cfg,
        tr_cfg=tr_cfg,
        struct_cfg=struct_cfg,
        step_cfg=step_cfg,
        dm_cfg=dm_cfg,
        robust_cfg=robust_cfg,
        preproc_cfg=preproc_cfg,
        gate_cfg=gate_cfg,
    )
