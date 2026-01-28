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

from pqc.config import BadMeasConfig, FeatureConfig, MergeConfig, StructureConfig, TransientConfig, StepConfig
from pqc.io.timfile import parse_all_timfiles
from pqc.io.libstempo_loader import load_libstempo
from pqc.io.merge import merge_time_and_meta
from pqc.features.backend_keys import ensure_sys_group
from pqc.features.feature_extraction import add_feature_columns
from pqc.detect.bad_measurements import detect_bad
from pqc.detect.feature_structure import detect_binned_structure, detrend_residuals_binned
from pqc.detect.transients import scan_transients
from pqc.detect.step_changes import detect_step, detect_dm_step
from pqc.utils.logging import info, warn

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
    dm_cfg: StepConfig = StepConfig(enabled=True, min_points=20, delta_chi2_thresh=25.0, scope="both"),
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
        drop_unmatched (bool): If True, drop TOAs whose metadata could not be
            matched.

    Returns:
        pandas.DataFrame: Timing, metadata, and QC annotations. The output
        includes the merged timfile metadata plus ``bad``, ``bad_day``, ``z``,
        ``transient_id``, ``transient_amp``, ``transient_t0``, and
        ``transient_delta_chi2``. If enabled, feature columns such as
        ``orbital_phase`` and ``solar_elongation_deg`` are added, plus
        structure-test summaries (``structure_*``) and optional
        ``resid_detrended``.

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
    out = []
    do_detrend = struct_cfg.mode in ("detrend", "both")
    do_test = struct_cfg.mode in ("test", "both")
    circ = set(struct_cfg.circular_features)
    structure_cols = list(struct_cfg.structure_group_cols or (backend_col,))
    structure_cols = [c for c in structure_cols if c in df.columns]
    if not structure_cols:
        structure_cols = [backend_col]

    df_work = df.copy()
    resid_col = "resid"
    if do_detrend:
        df_work["resid_detrended"] = df_work["resid"].to_numpy(dtype=float)
        resid_col = "resid_detrended"
        for _, sub in df_work.groupby(structure_cols):
            idx = sub.index
            for feat in struct_cfg.detrend_features:
                sub = detrend_residuals_binned(
                    sub,
                    feat,
                    resid_col=resid_col,
                    sigma_col="sigma",
                    nbins=struct_cfg.nbins,
                    circular=feat in circ,
                    min_per_bin=struct_cfg.min_per_bin,
                    out_col=resid_col,
                )
            df_work.loc[idx, resid_col] = sub[resid_col].to_numpy()

    if do_test:
        for _, sub in df_work.groupby(structure_cols):
            idx = sub.index
            for feat in struct_cfg.structure_features:
                base = f"structure_{feat}"
                if feat not in sub.columns:
                    df_work.loc[idx, f"{base}_chi2"] = np.nan
                    df_work.loc[idx, f"{base}_dof"] = 0
                    df_work.loc[idx, f"{base}_p"] = np.nan
                    df_work.loc[idx, f"{base}_present"] = False
                    continue
                res = detect_binned_structure(
                    sub,
                    feat,
                    resid_col=resid_col,
                    sigma_col="sigma",
                    nbins=struct_cfg.nbins,
                    circular=feat in circ,
                    min_per_bin=struct_cfg.min_per_bin,
                )
                p_like = res.get("p_like", np.nan)
                df_work.loc[idx, f"{base}_chi2"] = res.get("chi2", np.nan)
                df_work.loc[idx, f"{base}_dof"] = res.get("dof", 0)
                df_work.loc[idx, f"{base}_p"] = p_like
                df_work.loc[idx, f"{base}_present"] = bool(np.isfinite(p_like) and p_like < struct_cfg.p_thresh)

    # Optional step/DM step detection across all TOAs
    if step_cfg.enabled and step_cfg.scope in ("global", "both"):
        df_work = detect_step(
            df_work,
            mjd_col="mjd",
            resid_col=resid_col,
            sigma_col="sigma",
            min_points=step_cfg.min_points,
            delta_chi2_thresh=step_cfg.delta_chi2_thresh,
            prefix="step_global",
        )
    if dm_cfg.enabled and dm_cfg.scope in ("global", "both"):
        df_work = detect_dm_step(
            df_work,
            mjd_col="mjd",
            resid_col=resid_col,
            sigma_col="sigma",
            freq_col="freq",
            min_points=dm_cfg.min_points,
            delta_chi2_thresh=dm_cfg.delta_chi2_thresh,
            prefix="dm_step_global",
        )

    for key, sub in df_work.groupby(backend_col):
        sub1 = detect_bad(
            sub,
            tau_corr_days=bad_cfg.tau_corr_days,
            fdr_q=bad_cfg.fdr_q,
            mark_only_worst_per_day=bad_cfg.mark_only_worst_per_day,
            resid_col=resid_col,
        )
        sub2 = scan_transients(
            sub1,
            tau_rec_days=tr_cfg.tau_rec_days,
            window_mult=tr_cfg.window_mult,
            min_points=tr_cfg.min_points,
            delta_chi2_thresh=tr_cfg.delta_chi2_thresh,
            suppress_overlap=tr_cfg.suppress_overlap,
            resid_col=resid_col,
        )
        if step_cfg.enabled and step_cfg.scope in ("backend", "both"):
            sub2 = detect_step(
                sub2,
                mjd_col="mjd",
                resid_col=resid_col,
                sigma_col="sigma",
                min_points=step_cfg.min_points,
                delta_chi2_thresh=step_cfg.delta_chi2_thresh,
                prefix="step",
            )
        if dm_cfg.enabled and dm_cfg.scope in ("backend", "both"):
            sub2 = detect_dm_step(
                sub2,
                mjd_col="mjd",
                resid_col=resid_col,
                sigma_col="sigma",
                freq_col="freq",
                min_points=dm_cfg.min_points,
                delta_chi2_thresh=dm_cfg.delta_chi2_thresh,
                prefix="dm_step",
            )
        out.append(sub2)

    if not out:
        # All rows were dropped/unmatched, or no groups remained.
        return df.iloc[0:0].copy()

    df_out = pd.concat(out, axis=0).sort_values("mjd").reset_index(drop=True)
    return df_out
