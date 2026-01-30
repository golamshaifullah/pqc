"""Provide a command-line interface for the PQC pipeline.

This module wires CLI flags to :func:`pqc.pipeline.run_pipeline`. The CLI is
intentionally minimal and suitable for scripting or batch workflows where a
CSV output is desired. Optional feature extraction and structure diagnostics
are exposed via additional flags.

Examples:
    Basic usage from a module invocation:

    >>> # doctest: +SKIP
    >>> # python -m pqc.cli --par /data/J1909-3744.par --out out.csv

    Equivalent usage via the console script:

    >>> # doctest: +SKIP
    >>> # pqc --par /data/J1909-3744.par --out out.csv

See Also:
    pqc.pipeline.run_pipeline: Python API for the same workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pqc.config import (
    BadMeasConfig,
    FeatureConfig,
    MergeConfig,
    OutlierGateConfig,
    PreprocConfig,
    StepConfig,
    StructureConfig,
    TransientConfig,
)
from pqc.pipeline import run_pipeline
from pqc.utils.diagnostics import export_structure_table
from pqc.utils.logging import configure_logging


def _parse_csv_list(val: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated list into a tuple.

    Args:
        val (str | None): Input string or None.

    Returns:
        tuple[str, ...] | None: Parsed values, or None if ``val`` is None.
    """
    if val is None:
        return None
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return tuple(parts)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser for PTA QC options.
    """
    p = argparse.ArgumentParser(
        description="PTA residual QC: bad measurements + exponential transients"
    )

    p.add_argument(
        "--par", required=True, help="Path to pulsar .par file. Expects a sibling *_all.tim."
    )
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument(
        "--settings-out",
        default=None,
        help="Optional TOML path to write run settings (default: alongside --out, with .pqc_settings.toml suffix).",
    )
    p.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)."
    )
    p.add_argument("--log-format", default="%(message)s", help="Logging format string.")

    p.add_argument(
        "--backend-col",
        default="group",
        help="Column used to split backend groups (default: group).",
    )

    p.add_argument(
        "--tol-seconds",
        type=float,
        default=2.0,
        help="MJD merge tolerance in seconds (default: 2s).",
    )
    p.add_argument(
        "--drop-unmatched", action="store_true", help="Drop TOAs with unmatched metadata."
    )

    p.add_argument(
        "--tau-corr-min",
        type=float,
        default=30.0,
        help="OU correlation timescale in minutes (default: 30).",
    )
    p.add_argument(
        "--fdr-q", type=float, default=0.01, help="FDR q for day-level bad-measurement detection."
    )
    p.add_argument(
        "--mark-all-on-bad-days", action="store_true", help="Mark all TOAs on flagged days as bad."
    )

    p.add_argument(
        "--tau-rec-days", type=float, default=7.0, help="Transient recovery timescale (days)."
    )
    p.add_argument(
        "--window-mult", type=float, default=5.0, help="Window length = window_mult * tau_rec."
    )
    p.add_argument("--min-points", type=int, default=6, help="Minimum points in transient window.")
    p.add_argument(
        "--delta-chi2",
        type=float,
        default=25.0,
        help="Delta-chi2 threshold for transient detection.",
    )
    p.add_argument(
        "--event-instrument",
        action="store_true",
        help="Print per-event membership diagnostics (z_pt stats).",
    )
    p.add_argument(
        "--transient-scope",
        choices=["backend", "global", "both"],
        default="backend",
        help="Transient detection scope: backend/global/both (default: backend).",
    )

    p.add_argument(
        "--no-orbital-phase", action="store_true", help="Disable orbital phase feature extraction."
    )
    p.add_argument(
        "--no-solar-elongation",
        action="store_true",
        help="Disable solar elongation feature extraction.",
    )
    p.add_argument(
        "--add-elevation",
        action="store_true",
        help="Add elevation feature (requires astropy + telescope site).",
    )
    p.add_argument(
        "--add-airmass",
        action="store_true",
        help="Add airmass feature (requires astropy + telescope site).",
    )
    p.add_argument(
        "--add-parallactic-angle",
        action="store_true",
        help="Add parallactic angle feature (requires astropy + telescope site).",
    )
    p.add_argument(
        "--add-freq-bin", action="store_true", help="Add a linear frequency-bin index feature."
    )
    p.add_argument("--freq-bins", type=int, default=8, help="Number of frequency bins if enabled.")
    p.add_argument(
        "--observatory-file",
        default=None,
        help="Optional observatory XYZ file for telescope locations.",
    )

    p.add_argument(
        "--structure-mode",
        choices=["none", "detrend", "test", "both"],
        default="none",
        help="Feature-structure mode: none/detrend/test/both.",
    )
    p.add_argument(
        "--structure-detrend-features",
        default=None,
        help="Comma-separated feature columns to detrend against.",
    )
    p.add_argument(
        "--structure-test-features",
        default=None,
        help="Comma-separated feature columns to test for structure.",
    )
    p.add_argument(
        "--structure-circular-features",
        default=None,
        help="Comma-separated circular features (phase in [0,1)).",
    )
    p.add_argument(
        "--structure-group-cols",
        default=None,
        help="Comma-separated grouping columns for detrend/test (default: backend group).",
    )
    p.add_argument(
        "--structure-nbins",
        type=int,
        default=12,
        help="Number of bins for structure tests/detrending.",
    )
    p.add_argument("--structure-min-per-bin", type=int, default=3, help="Minimum points per bin.")
    p.add_argument(
        "--structure-p-thresh",
        type=float,
        default=0.01,
        help="P-value threshold for structure flag.",
    )
    p.add_argument(
        "--structure-summary-out",
        default=None,
        help="Optional CSV path for structure summary table.",
    )

    p.add_argument(
        "--detrend-features",
        default=None,
        help="Comma-separated feature columns to detrend for detectors.",
    )
    p.add_argument(
        "--rescale-feature",
        default=None,
        help="Feature column for variance rescaling before detectors.",
    )
    p.add_argument(
        "--condition-on",
        default=None,
        help="Comma-separated grouping columns for preprocessing (default: group).",
    )
    p.add_argument(
        "--use-preproc-for",
        default=None,
        help="Comma-separated detectors to use preprocessed residuals (ou,transient,mad,step,dmstep).",
    )
    p.add_argument(
        "--preproc-nbins",
        type=int,
        default=12,
        help="Number of bins for preprocessing mean/variance models.",
    )
    p.add_argument(
        "--preproc-min-per-bin",
        type=int,
        default=5,
        help="Minimum points per bin for preprocessing.",
    )
    p.add_argument(
        "--preproc-circular-features",
        default=None,
        help="Comma-separated circular features for preprocessing (phase in [0,1)).",
    )
    p.add_argument(
        "--outlier-gate", action="store_true", help="Enable hard sigma gate for outlier membership."
    )
    p.add_argument(
        "--outlier-gate-sigma",
        type=float,
        default=3.0,
        help="Sigma threshold for outlier gate (default: 3).",
    )
    p.add_argument(
        "--outlier-gate-resid-col", default=None, help="Residual column to gate on (default: auto)."
    )
    p.add_argument(
        "--outlier-gate-sigma-col", default=None, help="Sigma column to gate on (default: auto)."
    )

    return p


def main() -> None:
    """Run the QC pipeline from command-line arguments.

    This function parses CLI options, constructs configuration objects, runs
    the pipeline, and writes the output CSV. If requested, it also writes a
    structure-summary CSV derived from the per-group structure diagnostics.

    Raises:
        FileNotFoundError: If the ``.par`` file or the sibling ``*_all.tim`` is
            missing.

    Examples:
        Run with defaults and save to CSV:

        >>> # doctest: +SKIP
        >>> # pqc --par /data/J1909-3744.par --out out.csv
    """
    args = build_parser().parse_args()
    configure_logging(level=args.log_level, fmt=args.log_format)
    settings_out = args.settings_out
    if settings_out is None:
        out_path = Path(args.out)
        settings_out = out_path.with_suffix(".pqc_settings.toml")

    merge_cfg = MergeConfig(tol_days=args.tol_seconds / 86400.0)
    bad_cfg = BadMeasConfig(
        tau_corr_days=args.tau_corr_min / (60.0 * 24.0),
        fdr_q=args.fdr_q,
        mark_only_worst_per_day=(not args.mark_all_on_bad_days),
    )
    tr_cfg = TransientConfig(
        tau_rec_days=args.tau_rec_days,
        window_mult=args.window_mult,
        min_points=args.min_points,
        delta_chi2_thresh=args.delta_chi2,
        suppress_overlap=True,
        instrument=bool(args.event_instrument),
        scope=args.transient_scope,
    )
    step_cfg = StepConfig(instrument=bool(args.event_instrument))
    dm_cfg = StepConfig(instrument=bool(args.event_instrument))
    defaults = StructureConfig()
    detrend_feats = _parse_csv_list(args.structure_detrend_features) or defaults.detrend_features
    test_feats = _parse_csv_list(args.structure_test_features) or defaults.structure_features
    circ_feats = _parse_csv_list(args.structure_circular_features) or defaults.circular_features
    group_cols = _parse_csv_list(args.structure_group_cols)

    feature_cfg = FeatureConfig(
        add_orbital_phase=not args.no_orbital_phase,
        add_solar_elongation=not args.no_solar_elongation,
        add_elevation=args.add_elevation,
        add_airmass=args.add_airmass,
        add_parallactic_angle=args.add_parallactic_angle,
        add_freq_bin=args.add_freq_bin,
        freq_bins=args.freq_bins,
        observatory_path=args.observatory_file,
    )
    struct_cfg = StructureConfig(
        mode=args.structure_mode,
        detrend_features=detrend_feats,
        structure_features=test_feats,
        nbins=args.structure_nbins,
        min_per_bin=args.structure_min_per_bin,
        p_thresh=args.structure_p_thresh,
        circular_features=circ_feats,
        structure_group_cols=group_cols,
    )
    preproc_defaults = PreprocConfig()
    preproc_cfg = PreprocConfig(
        detrend_features=_parse_csv_list(args.detrend_features)
        or preproc_defaults.detrend_features,
        rescale_feature=args.rescale_feature,
        condition_on=_parse_csv_list(args.condition_on) or preproc_defaults.condition_on,
        use_preproc_for=_parse_csv_list(args.use_preproc_for) or preproc_defaults.use_preproc_for,
        nbins=args.preproc_nbins,
        min_per_bin=args.preproc_min_per_bin,
        circular_features=_parse_csv_list(args.preproc_circular_features)
        or preproc_defaults.circular_features,
    )
    gate_cfg = OutlierGateConfig(
        enabled=bool(args.outlier_gate),
        sigma_thresh=float(args.outlier_gate_sigma),
        resid_col=args.outlier_gate_resid_col,
        sigma_col=args.outlier_gate_sigma_col,
    )

    df = run_pipeline(
        args.par,
        backend_col=args.backend_col,
        bad_cfg=bad_cfg,
        tr_cfg=tr_cfg,
        merge_cfg=merge_cfg,
        feature_cfg=feature_cfg,
        struct_cfg=struct_cfg,
        step_cfg=step_cfg,
        dm_cfg=dm_cfg,
        preproc_cfg=preproc_cfg,
        gate_cfg=gate_cfg,
        drop_unmatched=args.drop_unmatched,
        settings_out=settings_out,
    )

    df.to_csv(args.out, index=False)
    if args.structure_summary_out:
        cols = struct_cfg.structure_group_cols or (args.backend_col,)
        summary = export_structure_table(df, group_cols=tuple(cols))
        summary.to_csv(args.structure_summary_out, index=False)


if __name__ == "__main__":
    main()
