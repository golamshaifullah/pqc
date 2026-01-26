"""Run the end-to-end PQC pipeline.

The primary entry point is :func:`run_pipeline`, which executes the following
steps:

1. Parse timfiles (including INCLUDE recursion).
2. Load libstempo arrays (TOA/residual/error/frequency).
3. Merge arrays with timfile metadata by nearest MJD.
4. Ensure backend keys (``sys``/``group``) exist.
5. Detect bad measurements and transient events per backend group.

See Also:
    pqc.io.timfile.parse_all_timfiles: Parse timfile metadata.
    pqc.io.libstempo_loader.load_libstempo: Load timing arrays via libstempo.
    pqc.io.merge.merge_time_and_meta: Merge timing arrays with metadata.
    pqc.features.backend_keys.ensure_sys_group: Normalize backend keys.
    pqc.detect.bad_measurements.detect_bad: Flag bad measurements.
    pqc.detect.transients.scan_transients: Detect exponential recoveries.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

from pqc.config import BadMeasConfig, TransientConfig, MergeConfig
from pqc.io.timfile import parse_all_timfiles
from pqc.io.libstempo_loader import load_libstempo
from pqc.io.merge import merge_time_and_meta
from pqc.features.backend_keys import ensure_sys_group
from pqc.detect.bad_measurements import detect_bad
from pqc.detect.transients import scan_transients
from pqc.utils.logging import info, warn

def run_pipeline(parfile: str | Path,
                 *,
                 backend_col: str = "group",
                 bad_cfg: BadMeasConfig = BadMeasConfig(),
                 tr_cfg: TransientConfig = TransientConfig(),
                 merge_cfg: MergeConfig = MergeConfig(),
                 drop_unmatched: bool = False) -> pd.DataFrame:
    """Run the full PTA QC pipeline for a single pulsar.

    This function reads the ``.par`` file and its sibling ``*_all.tim``,
    constructs timing and metadata tables, and annotates each TOA with QC
    flags and transient detections.

    Args:
        parfile: Path to the pulsar ``.par`` file. A sibling ``*_all.tim`` is
            required and will be discovered by filename convention.
        backend_col: Column used to group TOAs for per-backend QC.
        bad_cfg: Configuration for bad-measurement detection.
        tr_cfg: Configuration for transient detection.
        merge_cfg: Configuration for timfile/TOA matching.
        drop_unmatched: If True, drop TOAs whose metadata could not be matched.

    Returns:
        A DataFrame containing timing, metadata, and QC annotations. The
        output includes the merged timfile metadata plus the columns
        ``bad``, ``bad_day``, ``z``, ``transient_id``, ``transient_amp``,
        ``transient_t0``, and ``transient_delta_chi2``.

    Raises:
        FileNotFoundError: If ``parfile`` or the matching ``*_all.tim`` is missing.

    Notes:
        This function logs progress via :mod:`pqc.utils.logging`. It does not
        mutate input files, but it will run libstempo, which may be sensitive
        to local tempo2 installation details.

    Examples:
        Run with defaults::

            from pqc.pipeline import run_pipeline
            df = run_pipeline("/data/J1909-3744.par")

        Run with custom detection parameters::

            from pqc.config import BadMeasConfig, TransientConfig
            df = run_pipeline(
                "/data/J1909-3744.par",
                bad_cfg=BadMeasConfig(tau_corr_days=0.03),
                tr_cfg=TransientConfig(delta_chi2_thresh=40.0),
            )
    """
    parfile = Path(parfile)
    if not parfile.exists():
        raise FileNotFoundError(str(parfile))

    all_tim = str(parfile).replace(".par", "_all.tim")
    if not Path(all_tim).exists():
        raise FileNotFoundError(all_tim)

    info("[1/5] Parse timfiles")
    tim_res = parse_all_timfiles(all_tim)
    df_tim = tim_res.df
    if tim_res.dropped_lines:
        warn(f"Dropped {tim_res.dropped_lines} malformed TOA lines while parsing timfiles.")

    info("[2/5] Load libstempo")
    df_time = load_libstempo(parfile)

    info("[3/5] Merge by MJD")
    df = merge_time_and_meta(df_time, df_tim, tol_days=merge_cfg.tol_days)

    n_unmatched = int(df["filename"].isna().sum())
    if n_unmatched:
        warn(
            f"{n_unmatched} TOAs unmatched to tim metadata (within tol={merge_cfg.tol_days} days). "
            + ("Dropping them." if drop_unmatched else "Keeping them with NaN metadata.")
        )
        if drop_unmatched:
            df = df.loc[~df["filename"].isna()].reset_index(drop=True)

    info("[4/5] Ensure sys/group")
    df = ensure_sys_group(df)

    info("[5/5] Detect bad measurements + transients per backend")
    out = []
    for key, sub in df.groupby(backend_col):
        sub1 = detect_bad(
            sub,
            tau_corr_days=bad_cfg.tau_corr_days,
            fdr_q=bad_cfg.fdr_q,
            mark_only_worst_per_day=bad_cfg.mark_only_worst_per_day,
        )
        sub2 = scan_transients(
            sub1,
            tau_rec_days=tr_cfg.tau_rec_days,
            window_mult=tr_cfg.window_mult,
            min_points=tr_cfg.min_points,
            delta_chi2_thresh=tr_cfg.delta_chi2_thresh,
            suppress_overlap=tr_cfg.suppress_overlap,
        )
        out.append(sub2)

    if not out:
        # All rows were dropped/unmatched, or no groups remained.
        return df.iloc[0:0].copy()

    df_out = pd.concat(out, axis=0).sort_values("mjd").reset_index(drop=True)
    return df_out
