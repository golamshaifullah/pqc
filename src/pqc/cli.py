"""Run the PQC pipeline from a command-line interface.

This module wires configuration flags to :func:`pqc.pipeline.run_pipeline`.
It is intentionally minimal and suitable for scripting or batch workflows
where a CSV output is desired.

Examples:
    Basic usage from a module invocation::

        python -m pqc.cli --par /data/J1909-3744.par --out out.csv

    Equivalent usage via the console script::

        pqc --par /data/J1909-3744.par --out out.csv

See Also:
    pqc.pipeline.run_pipeline: Python API for the same workflow.
"""

from __future__ import annotations
import argparse
from pqc.pipeline import run_pipeline
from pqc.config import BadMeasConfig, TransientConfig, MergeConfig

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        An :class:`argparse.ArgumentParser` configured for PTA QC options.
    """
    p = argparse.ArgumentParser(description="PTA residual QC: bad measurements + exponential transients")

    p.add_argument("--par", required=True, help="Path to pulsar .par file. Expects a sibling *_all.tim.")
    p.add_argument("--out", required=True, help="Output CSV path.")

    p.add_argument("--backend-col", default="group", help="Column used to split backend groups (default: group).")

    p.add_argument("--tol-seconds", type=float, default=2.0, help="MJD merge tolerance in seconds (default: 2s).")
    p.add_argument("--drop-unmatched", action="store_true", help="Drop TOAs with unmatched metadata.")

    p.add_argument("--tau-corr-min", type=float, default=30.0, help="OU correlation timescale in minutes (default: 30)." )
    p.add_argument("--fdr-q", type=float, default=0.01, help="FDR q for day-level bad-measurement detection." )
    p.add_argument("--mark-all-on-bad-days", action="store_true", help="Mark all TOAs on flagged days as bad.")

    p.add_argument("--tau-rec-days", type=float, default=7.0, help="Transient recovery timescale (days)." )
    p.add_argument("--window-mult", type=float, default=5.0, help="Window length = window_mult * tau_rec." )
    p.add_argument("--min-points", type=int, default=6, help="Minimum points in transient window." )
    p.add_argument("--delta-chi2", type=float, default=25.0, help="Delta-chi2 threshold for transient detection." )

    return p

def main() -> None:
    """Run the QC pipeline from command-line arguments.

    This function parses CLI options, constructs configuration objects, runs
    the pipeline, and writes the output CSV.

    Raises:
        FileNotFoundError: If the ``.par`` file or the sibling ``*_all.tim`` is
            missing.

    Examples:
        Run with defaults and save to CSV::

            pqc --par /data/J1909-3744.par --out out.csv
    """
    args = build_parser().parse_args()

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
    )

    df = run_pipeline(
        args.par,
        backend_col=args.backend_col,
        bad_cfg=bad_cfg,
        tr_cfg=tr_cfg,
        merge_cfg=merge_cfg,
        drop_unmatched=args.drop_unmatched,
    )

    df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
