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
    SolarCutConfig,
    OrbitalPhaseCutConfig,
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
from pqc.utils.settings import write_run_settings_toml

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
    def _apply_grouped_global_step(
        frame: pd.DataFrame,
        *,
        prefix: str,
        resid_col: str,
        sigma_col: str,
        group_col: str,
        member_eta: float,
        member_tmax_days: float | None,
        min_points: int,
        freq_col: str | None = None,
    ) -> pd.DataFrame:
        if f"{prefix}_t0" not in frame.columns or group_col not in frame.columns:
            return frame
        t0_vals = pd.to_numeric(frame[f"{prefix}_t0"], errors="coerce")
        if not np.isfinite(t0_vals).any():
            return frame
        t0 = float(t0_vals.dropna().iloc[0])
        t_all = pd.to_numeric(frame["mjd"], errors="coerce").to_numpy(dtype=float)
        t_end = np.nanmax(t_all) if member_tmax_days is None else (t0 + float(member_tmax_days))

        for g, sub in frame.groupby(group_col):
            idx = sub.index
            t = pd.to_numeric(sub["mjd"], errors="coerce").to_numpy(dtype=float)
            r = pd.to_numeric(sub[resid_col], errors="coerce").to_numpy(dtype=float)
            s = pd.to_numeric(sub[sigma_col], errors="coerce").to_numpy(dtype=float)

            pre = np.isfinite(t) & (t < t0) & np.isfinite(r) & np.isfinite(s) & (s > 0)
            post = (
                np.isfinite(t)
                & (t >= t0)
                & (t <= t_end)
                & np.isfinite(r)
                & np.isfinite(s)
                & (s > 0)
            )
            if freq_col is not None and freq_col in sub.columns:
                f = pd.to_numeric(sub[freq_col], errors="coerce").to_numpy(dtype=float)
                goodf = np.isfinite(f) & (f != 0)
                pre &= goodf
                post &= goodf
                y_pre = r[pre] * (f[pre] ** 2)
                y_post = r[post] * (f[post] ** 2)
                s_pre = s[pre] * (f[pre] ** 2)
                s_post = s[post] * (f[post] ** 2)
            else:
                y_pre, y_post = r[pre], r[post]
                s_pre, s_post = s[pre], s[post]

            if np.count_nonzero(pre) < min_points or np.count_nonzero(post) < min_points:
                frame.loc[idx, f"{prefix}_id"] = -1
                frame.loc[idx, f"{prefix}_applicable"] = False
                frame.loc[idx, f"{prefix}_informative"] = False
                frame.loc[idx, f"{prefix}_amp"] = np.nan
                frame.loc[idx, f"{prefix}_n_applicable"] = 0
                frame.loc[idx, f"{prefix}_n_informative"] = 0
                frame.loc[idx, f"{prefix}_n_members"] = 0
                frame.loc[idx, f"{prefix}_frac_informative_z_lt1"] = 0.0
                frame.loc[idx, f"{prefix}_frac_z_lt1"] = 0.0
                frame.loc[idx, f"{prefix}_z_min"] = np.nan
                frame.loc[idx, f"{prefix}_z_med"] = np.nan
                frame.loc[idx, f"{prefix}_z_max"] = np.nan
                continue

            w_pre = 1.0 / (s_pre**2)
            w_post = 1.0 / (s_post**2)
            mu_pre = np.sum(w_pre * y_pre) / np.sum(w_pre)
            mu_post = np.sum(w_post * y_post) / np.sum(w_post)
            amp = float(mu_post - mu_pre)

            applicable = np.isfinite(t) & (t >= t0) & (t <= t_end)
            z_pt = np.full_like(t, np.nan, dtype=float)
            if freq_col is not None and freq_col in sub.columns:
                f = pd.to_numeric(sub[freq_col], errors="coerce").to_numpy(dtype=float)
                good = applicable & np.isfinite(s) & (s > 0) & np.isfinite(f) & (f != 0)
                effect = np.zeros_like(t, dtype=float)
                effect[good] = np.abs(amp) / (f[good] ** 2)
            else:
                good = applicable & np.isfinite(s) & (s > 0)
                effect = np.zeros_like(t, dtype=float)
                effect[good] = np.abs(amp)
            z_pt[good] = effect[good] / s[good]
            informative = applicable.copy()
            if np.isfinite(member_eta):
                informative &= z_pt > float(member_eta)

            frame.loc[idx, f"{prefix}_id"] = -1
            frame.loc[idx[applicable], f"{prefix}_id"] = 0
            frame.loc[idx, f"{prefix}_applicable"] = applicable
            frame.loc[idx, f"{prefix}_informative"] = informative
            frame.loc[idx, f"{prefix}_amp"] = amp
            z_inf = z_pt[informative]
            frame.loc[idx, f"{prefix}_n_applicable"] = int(np.count_nonzero(applicable))
            frame.loc[idx, f"{prefix}_n_informative"] = int(np.count_nonzero(informative))
            frame.loc[idx, f"{prefix}_n_members"] = int(np.count_nonzero(informative))
            if len(z_inf):
                frame.loc[idx, f"{prefix}_frac_informative_z_lt1"] = float(np.mean(z_inf < 1.0))
                frame.loc[idx, f"{prefix}_frac_z_lt1"] = float(np.mean(z_inf < 1.0))
                frame.loc[idx, f"{prefix}_z_min"] = float(np.nanmin(z_inf))
                frame.loc[idx, f"{prefix}_z_med"] = float(np.nanmedian(z_inf))
                frame.loc[idx, f"{prefix}_z_max"] = float(np.nanmax(z_inf))
            else:
                frame.loc[idx, f"{prefix}_frac_informative_z_lt1"] = 0.0
                frame.loc[idx, f"{prefix}_frac_z_lt1"] = 0.0
                frame.loc[idx, f"{prefix}_z_min"] = np.nan
                frame.loc[idx, f"{prefix}_z_med"] = np.nan
                frame.loc[idx, f"{prefix}_z_max"] = np.nan

        return frame

    def _apply_grouped_global_transients(
        frame: pd.DataFrame,
        *,
        resid_col: str,
        sigma_col: str,
        group_col: str,
        tau_rec_days: float,
        window_mult: float,
        member_eta: float,
    ) -> pd.DataFrame:
        if "transient_global_id" not in frame.columns or group_col not in frame.columns:
            return frame

        events = frame.loc[
            frame["transient_global_id"] >= 0,
            ["transient_global_id", "transient_global_t0", "transient_global_delta_chi2"],
        ]
        if events.empty:
            return frame

        events = events.dropna(
            subset=["transient_global_id", "transient_global_t0"]
        ).drop_duplicates("transient_global_id")
        if events.empty:
            return frame

        frame["transient_global_id"] = -1
        frame["transient_global_amp"] = np.nan
        frame["transient_global_t0"] = np.nan
        frame["transient_global_delta_chi2"] = np.nan
        frame["transient_global_n_members"] = 0
        frame["transient_global_frac_z_lt1"] = 0.0
        frame["transient_global_z_min"] = np.nan
        frame["transient_global_z_med"] = np.nan
        frame["transient_global_z_max"] = np.nan

        for _, ev in events.iterrows():
            tid = int(ev["transient_global_id"])
            t0 = float(ev["transient_global_t0"])
            delta = (
                float(ev["transient_global_delta_chi2"])
                if np.isfinite(ev.get("transient_global_delta_chi2", np.nan))
                else np.nan
            )
            w_end = window_mult * tau_rec_days

            for g, sub in frame.groupby(group_col):
                idx = sub.index
                t = pd.to_numeric(sub["mjd"], errors="coerce").to_numpy(dtype=float)
                r = pd.to_numeric(sub[resid_col], errors="coerce").to_numpy(dtype=float)
                s = pd.to_numeric(sub[sigma_col], errors="coerce").to_numpy(dtype=float)

                in_win = np.isfinite(t) & (t >= t0) & (t <= t0 + w_end)
                good = in_win & np.isfinite(r) & np.isfinite(s) & (s > 0)
                if np.count_nonzero(good) < 2:
                    continue

                tt = t[good] - t0
                f = np.exp(-tt / tau_rec_days)
                w = 1.0 / (s[good] ** 2)
                denom = np.sum(w * f * f)
                if denom <= 0:
                    continue
                A = np.sum(w * f * r[good]) / denom

                z_pt = np.full_like(t, np.nan, dtype=float)
                model = np.zeros_like(t, dtype=float)
                model[in_win] = A * np.exp(-(t[in_win] - t0) / tau_rec_days)
                z_good = in_win & np.isfinite(s) & (s > 0)
                z_pt[z_good] = np.abs(model[z_good]) / s[z_good]

                member = in_win.copy()
                if np.isfinite(member_eta):
                    member &= z_pt > float(member_eta)

                if not np.any(member):
                    continue

                frame.loc[idx[member], "transient_global_id"] = tid
                frame.loc[idx[member], "transient_global_amp"] = A
                frame.loc[idx[member], "transient_global_t0"] = t0
                frame.loc[idx[member], "transient_global_delta_chi2"] = delta

                z_mem = z_pt[member]
                frame.loc[idx[member], "transient_global_n_members"] = int(np.count_nonzero(member))
                frame.loc[idx[member], "transient_global_frac_z_lt1"] = (
                    float(np.mean(z_mem < 1.0)) if len(z_mem) else 0.0
                )
                frame.loc[idx[member], "transient_global_z_min"] = (
                    float(np.nanmin(z_mem)) if len(z_mem) else np.nan
                )
                frame.loc[idx[member], "transient_global_z_med"] = (
                    float(np.nanmedian(z_mem)) if len(z_mem) else np.nan
                )
                frame.loc[idx[member], "transient_global_z_max"] = (
                    float(np.nanmax(z_mem)) if len(z_mem) else np.nan
                )

        return frame

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
        df_work["preproc_notes"] = (
            f"group_cols={group_cols or (backend_col,)}; nbins={preproc_cfg.nbins}; min_per_bin={preproc_cfg.min_per_bin}"
        )

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
        df_work = _apply_grouped_global_step(
            df_work,
            prefix="step_global",
            resid_col=step_resid,
            sigma_col=step_sigma,
            group_col=backend_col,
            member_eta=step_cfg.member_eta,
            member_tmax_days=step_cfg.member_tmax_days,
            min_points=step_cfg.min_points,
            freq_col=None,
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
        df_work = _apply_grouped_global_step(
            df_work,
            prefix="dm_step_global",
            resid_col=dm_resid,
            sigma_col=dm_sigma,
            group_col=backend_col,
            member_eta=dm_cfg.member_eta,
            member_tmax_days=dm_cfg.member_tmax_days,
            min_points=dm_cfg.min_points,
            freq_col="freq",
        )

    if tr_cfg.scope in ("global", "both"):
        df_glob = scan_transients(
            df_work,
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
        rename_map = {
            "transient_id": "transient_global_id",
            "transient_amp": "transient_global_amp",
            "transient_t0": "transient_global_t0",
            "transient_delta_chi2": "transient_global_delta_chi2",
        }
        for src, dst in rename_map.items():
            if src in df_glob.columns:
                df_work[dst] = df_glob[src]
        df_work = _apply_grouped_global_transients(
            df_work,
            resid_col=tr_resid,
            sigma_col=tr_sigma,
            group_col=backend_col,
            tau_rec_days=tr_cfg.tau_rec_days,
            window_mult=tr_cfg.window_mult,
            member_eta=tr_cfg.member_eta,
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
    if "step_applicable" not in df_out.columns and "step_global_applicable" in df_out.columns:
        df_out["step_applicable"] = df_out["step_global_applicable"].fillna(False)
    if "step_informative" not in df_out.columns and "step_global_informative" in df_out.columns:
        df_out["step_informative"] = df_out["step_global_informative"].fillna(False)
    if "dm_step_applicable" not in df_out.columns and "dm_step_global_applicable" in df_out.columns:
        df_out["dm_step_applicable"] = df_out["dm_step_global_applicable"].fillna(False)
    if (
        "dm_step_informative" not in df_out.columns
        and "dm_step_global_informative" in df_out.columns
    ):
        df_out["dm_step_informative"] = df_out["dm_step_global_informative"].fillna(False)
    if "step_applicable" not in df_out.columns:
        df_out["step_applicable"] = False
    if "step_informative" not in df_out.columns:
        df_out["step_informative"] = False
    if "dm_step_applicable" not in df_out.columns:
        df_out["dm_step_applicable"] = False
    if "dm_step_informative" not in df_out.columns:
        df_out["dm_step_informative"] = False

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
            if mad_resid != resid_gate_col:
                warn(
                    f"Outlier gate uses {resid_gate_col}/{sigma_gate_col} but MAD uses {mad_resid}/sigma; set gate columns explicitly to align."
                )

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
    if "transient_global_id" in df_out.columns:
        df_out["event_member"] |= df_out["transient_global_id"].fillna(-1).to_numpy() >= 0
    if "step_informative" in df_out.columns:
        df_out["event_member"] |= df_out["step_informative"].fillna(False).to_numpy()
    if "dm_step_informative" in df_out.columns:
        df_out["event_member"] |= df_out["dm_step_informative"].fillna(False).to_numpy()
    df_out["event_member"] &= ~df_out["bad_point"].fillna(False)

    df_out["outlier_any"] = False
    df_out["outlier_any"] |= df_out["bad_point"]
    df_out["outlier_any"] |= df_out["event_member"]

    if solar_cfg.enabled:
        if "solar_elongation_deg" not in df_out.columns:
            warn("solar elongation not available; solar cut disabled for this run.")
        else:
            resid_col = ou_resid
            sigma_col = ou_sigma
            elong = pd.to_numeric(df_out["solar_elongation_deg"], errors="coerce").to_numpy(
                dtype=float
            )
            r = pd.to_numeric(df_out[resid_col], errors="coerce").to_numpy(dtype=float)
            s = pd.to_numeric(df_out[sigma_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(elong) & np.isfinite(r) & np.isfinite(s) & (s > 0)

            cut = None
            if solar_cfg.limit_deg is not None:
                cut = float(solar_cfg.limit_deg)
            elif valid.sum() >= int(solar_cfg.min_points):
                nbins = int(solar_cfg.nbins)
                edges = np.linspace(0.0, 180.0, nbins + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                z = np.abs(r[valid]) / s[valid]
                x = elong[valid]
                med = np.full(nbins, np.nan)
                for i in range(nbins):
                    mask = (x >= edges[i]) & (x < edges[i + 1])
                    if mask.any():
                        med[i] = np.nanmedian(z[mask])
                thresh = float(solar_cfg.sigma_thresh)
                valid_bins = np.isfinite(med)
                for i in range(nbins):
                    if not valid_bins[i]:
                        continue
                    tail = med[i:][valid_bins[i:]]
                    if tail.size > 0 and np.all(tail <= thresh):
                        cut = float(centers[i])
                        break
                if cut is None and np.any(med <= thresh):
                    cut = float(np.nanmean(centers[med <= thresh]))

            df_out["solar_cut_deg"] = np.nan if cut is None else float(cut)
            df_out["solar_cut_sigma_thresh"] = float(solar_cfg.sigma_thresh)
            df_out["solar_bad"] = False
            if cut is not None:
                df_out["solar_bad"] = pd.to_numeric(
                    df_out["solar_elongation_deg"], errors="coerce"
                ).to_numpy(dtype=float) <= float(cut)

    if orbital_cfg.enabled:
        if "orbital_phase" not in df_out.columns:
            warn("orbital phase not available; orbital phase cut disabled for this run.")
        else:
            resid_col = ou_resid
            sigma_col = ou_sigma
            phase = pd.to_numeric(df_out["orbital_phase"], errors="coerce").to_numpy(dtype=float)
            r = pd.to_numeric(df_out[resid_col], errors="coerce").to_numpy(dtype=float)
            s = pd.to_numeric(df_out[sigma_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(phase) & np.isfinite(r) & np.isfinite(s) & (s > 0)

            cut = None
            center = float(orbital_cfg.center_phase)
            if orbital_cfg.limit_phase is not None:
                cut = float(orbital_cfg.limit_phase)
            elif valid.sum() >= int(orbital_cfg.min_points):
                nbins = int(orbital_cfg.nbins)
                # distance to eclipse center, folded to [0, 0.5]
                dist = np.minimum(
                    np.abs(phase[valid] - center), 1.0 - np.abs(phase[valid] - center)
                )
                edges = np.linspace(0.0, 0.5, nbins + 1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                z = np.abs(r[valid]) / s[valid]
                med = np.full(nbins, np.nan)
                for i in range(nbins):
                    mask = (dist >= edges[i]) & (dist < edges[i + 1])
                    if mask.any():
                        med[i] = np.nanmedian(z[mask])
                thresh = float(orbital_cfg.sigma_thresh)
                valid_bins = np.isfinite(med)
                for i in range(nbins):
                    if not valid_bins[i]:
                        continue
                    tail = med[i:][valid_bins[i:]]
                    if tail.size > 0 and np.all(tail <= thresh):
                        cut = float(centers[i])
                        break
                if cut is None and np.any(med <= thresh):
                    cut = float(np.nanmean(centers[med <= thresh]))

            df_out["orbital_cut_phase"] = np.nan if cut is None else float(cut)
            df_out["orbital_cut_sigma_thresh"] = float(orbital_cfg.sigma_thresh)
            df_out["orbital_phase_bad"] = False
            if cut is not None:
                dist_all = np.minimum(np.abs(phase - center), 1.0 - np.abs(phase - center))
                df_out["orbital_phase_bad"] = np.isfinite(dist_all) & (dist_all <= float(cut))

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
    solar_cfg: SolarCutConfig = SolarCutConfig(),
    orbital_cfg: OrbitalPhaseCutConfig = OrbitalPhaseCutConfig(),
    drop_unmatched: bool = False,
    settings_out: str | Path | None = None,
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
        settings_out (str | Path | None): Optional TOML path to write the
            pipeline settings used for this run. If None, a sibling
            ``<parfile>/.pqc_settings.toml`` is created.

    Returns:
        pandas.DataFrame: Timing, metadata, and QC annotations. The output
        includes the merged timfile metadata plus ``bad``, ``bad_day``, ``z``,
        ``bad_ou``, ``bad_mad``, ``bad_point``, ``event_member``,
        ``transient_id``, ``step_id``, ``dm_step_id``,
        ``step_applicable``, ``step_informative``, ``dm_step_applicable``,
        ``dm_step_informative``, and ``outlier_any``
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

    if settings_out is None:
        settings_out = parfile.with_suffix(".pqc_settings.toml")
    write_run_settings_toml(
        settings_out,
        parfile=parfile,
        backend_col=backend_col,
        drop_unmatched=drop_unmatched,
        bad_cfg=bad_cfg,
        tr_cfg=tr_cfg,
        merge_cfg=merge_cfg,
        feature_cfg=feature_cfg,
        struct_cfg=struct_cfg,
        step_cfg=step_cfg,
        dm_cfg=dm_cfg,
        robust_cfg=robust_cfg,
        preproc_cfg=preproc_cfg,
        gate_cfg=gate_cfg,
        solar_cfg=solar_cfg,
        orbital_cfg=orbital_cfg,
    )

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
