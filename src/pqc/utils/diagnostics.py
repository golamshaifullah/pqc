"""Provide diagnostics helpers for QC outputs.

Summaries are printed in a human-readable form. Plotting is intentionally
kept out of this module to keep dependencies minimal.

See Also:
    pqc.utils.logging: Logging helpers used for summaries.
    pqc.pipeline.run_pipeline: Produces the DataFrame consumed by these helpers.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pqc.utils.logging import info, warn

def summarize_dataset(df: pd.DataFrame, backend_col: str = "group") -> None:
    """Print a compact summary of dataset composition.

    Args:
        df (pandas.DataFrame): QC DataFrame to summarize.
        backend_col (str): Column used to define backend groupings.

    Notes:
        This function prints to stdout/stderr via :mod:`pqc.utils.logging`.

    Examples:
        >>> import pandas as pd
        >>> summarize_dataset(pd.DataFrame({"mjd": [1.0, 2.0], "group": ["A", "B"]}))
    """
    info(f"Rows: {len(df)}")
    info(f"Columns: {len(df.columns)}")

    if "filename" in df.columns:
        n_unmatched = int(df["filename"].isna().sum())
        if n_unmatched:
            warn(f"Unmatched metadata rows (filename is NaN): {n_unmatched}")
        else:
            info("All rows have filename metadata.")

    if backend_col in df.columns:
        vc = df[backend_col].astype(str).value_counts()
        info(f"Backends ({backend_col}) count: {len(vc)}")
        info("Top 20 backends:")
        info(vc.head(20).to_string())

def summarize_results(df: pd.DataFrame, backend_col: str = "group") -> None:
    """Print summary of bad-measurement and transient outputs.

    Args:
        df (pandas.DataFrame): QC DataFrame containing annotation columns.
        backend_col (str): Column used to define backend groupings.

    Notes:
        ``bad_day`` counts TOAs labeled as bad-day members, not unique days.

    Examples:
        >>> import pandas as pd
        >>> summarize_results(pd.DataFrame({"bad": [True, False], "bad_day": [True, False]}))
    """
    if "bad" in df.columns:
        info(f"Bad TOAs: {int(df['bad'].fillna(False).sum())}")
    if "bad_day" in df.columns:
        info(f"Bad days: {int(df['bad_day'].fillna(False).sum())}")  # counts TOAs, not unique days

    if "transient_id" in df.columns:
        n_events = int(df["transient_id"].max() + 1) if len(df) and df["transient_id"].max() >= 0 else 0
        info(f"Transient events detected (per-backend ids): {n_events}")
        if n_events:
            ev = df[df["transient_id"] >= 0].copy()
            cols = [backend_col, "transient_id", "transient_t0", "transient_amp", "transient_delta_chi2"]
            cols = [c for c in cols if c in ev.columns]
            # one row per (backend, event)
            ev = ev.groupby([backend_col, "transient_id"], as_index=False).first()
            info("Detected events (first 30):")
            info(ev[cols].head(30).to_string(index=False))

def export_event_table(df: pd.DataFrame, backend_col: str = "group") -> pd.DataFrame:
    """Return a tidy event table (one row per detected transient).

    Args:
        df (pandas.DataFrame): QC DataFrame containing transient annotation
            columns.
        backend_col (str): Column used to define backend groupings.

    Returns:
        pandas.DataFrame: One row per detected transient event.

    Examples:
        >>> import pandas as pd
        >>> export_event_table(pd.DataFrame({"transient_id": [-1, 0]})).shape[0] >= 0
        True
    """
    if "transient_id" not in df.columns:
        return pd.DataFrame()
    ev = df[df["transient_id"] >= 0].copy()
    if ev.empty:
        return pd.DataFrame()
    ev = ev.groupby([backend_col, "transient_id"], as_index=False).first()
    return ev[[c for c in [backend_col, "transient_id", "transient_t0", "transient_amp", "transient_delta_chi2"] if c in ev.columns]]

def export_structure_table(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("group",),
    prefix: str = "structure_",
) -> pd.DataFrame:
    """Return a tidy table of feature-structure diagnostics per group.

    Args:
        df (pandas.DataFrame): QC DataFrame containing structure columns.
        group_cols (tuple[str, ...]): Columns defining structure groups.
        prefix (str): Prefix used for structure columns.

    Returns:
        pandas.DataFrame: One row per (group, feature) containing
        ``chi2``, ``dof``, ``p``, and ``present``.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "group": ["A", "A"],
        ...     "structure_orbital_phase_chi2": [1.0, 1.0],
        ...     "structure_orbital_phase_dof": [2, 2],
        ...     "structure_orbital_phase_p": [0.3, 0.3],
        ...     "structure_orbital_phase_present": [False, False],
        ... })
        >>> export_structure_table(df, group_cols=("group",)).shape[0]
        1
    """
    if df.empty:
        return pd.DataFrame()
    chi2_cols = [c for c in df.columns if c.startswith(prefix) and c.endswith("_chi2")]
    if not chi2_cols:
        return pd.DataFrame()

    features = [c[len(prefix):-len("_chi2")] for c in chi2_cols]
    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        cols = []

    rows = []
    grouped = df.groupby(cols, dropna=False) if cols else [((), df)]
    for key, sub in grouped:
        first = sub.iloc[0]
        key_vals = key if isinstance(key, tuple) else (key,)
        for feat in features:
            base = f"{prefix}{feat}"
            row = {"feature": feat}
            if cols:
                row.update({col: val for col, val in zip(cols, key_vals)})
            row["chi2"] = first.get(f"{base}_chi2", np.nan)
            row["dof"] = first.get(f"{base}_dof", np.nan)
            row["p"] = first.get(f"{base}_p", np.nan)
            row["present"] = bool(first.get(f"{base}_present", False))
            rows.append(row)

    return pd.DataFrame(rows)
