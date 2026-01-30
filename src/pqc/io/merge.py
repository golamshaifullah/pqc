"""Merge timing arrays with timfile metadata.

This module provides a small helper to join libstempo timing arrays with
parsed timfile metadata using nearest-neighbor matching in MJD space.

See Also:
    pqc.io.timfile.parse_all_timfiles: Parse timfile metadata.
    pqc.io.libstempo_loader.load_libstempo: Load timing arrays via libstempo.
"""

from __future__ import annotations
import pandas as pd


def merge_time_and_meta(
    df_time: pd.DataFrame,
    df_meta: pd.DataFrame,
    tol_days: float,
) -> pd.DataFrame:
    """Merge TOA arrays with timfile metadata using nearest-neighbor MJD.

    Args:
        df_time (pandas.DataFrame): Output of
            :func:`pqc.io.libstempo_loader.load_libstempo`.
        df_meta (pandas.DataFrame): Output of
            :func:`pqc.io.timfile.parse_all_timfiles`.
        tol_days (float): Maximum |ΔMJD| tolerance for matching.

    Returns:
        pandas.DataFrame: Tim metadata columns merged onto timing arrays.

    Notes:
        This uses :func:`pandas.merge_asof` with ``direction="nearest"``. Rows
        that cannot be matched within ``tol_days`` retain ``NaN`` metadata
        columns.

    Examples:
        >>> import pandas as pd
        >>> df_time = pd.DataFrame({"mjd": [1.0, 2.0]})
        >>> df_meta = pd.DataFrame({"mjd": [1.0], "filename": ["a.tim"]})
        >>> merge_time_and_meta(df_time, df_meta, tol_days=0.1).shape[0]
        2
    """
    dt = df_time.copy()
    dm = df_meta.copy()
    dt["mjd"] = dt["mjd"].astype("float64")
    dm["mjd"] = dm["mjd"].astype("float64")
    dt = dt.sort_values("mjd").reset_index(drop=True)
    dm = dm.sort_values("mjd").reset_index(drop=True)

    merged = pd.merge_asof(
        dt,
        dm,
        on="mjd",
        direction="nearest",
        tolerance=float(tol_days),
        suffixes=("", "_meta"),
    )
    return merged
