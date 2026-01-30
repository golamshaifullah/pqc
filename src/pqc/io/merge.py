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
    freq_tol_mhz: float | None = None,
) -> pd.DataFrame:
    """Merge TOA arrays with timfile metadata using nearest-neighbor MJD.

    Args:
        df_time (pandas.DataFrame): Output of
            :func:`pqc.io.libstempo_loader.load_libstempo`.
        df_meta (pandas.DataFrame): Output of
            :func:`pqc.io.timfile.parse_all_timfiles`.
        tol_days (float): Maximum |ΔMJD| tolerance for matching.
        freq_tol_mhz (float | None): Optional frequency tolerance (MHz) used to
            refine matching when MJD-only matching fails.

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
    if (
        freq_tol_mhz is not None
        and "_timfile" in merged.columns
        and "freq" in dt.columns
        and "freq" in dm.columns
    ):
        mask_unmatched = merged["_timfile"].isna()
        if mask_unmatched.any():
            dt2 = dt.loc[mask_unmatched].copy()
            dm2 = dm.copy()
            # Only attempt freq-based matching when frequency is present.
            dt2 = dt2.loc[dt2["freq"].notna() & dt2["mjd"].notna()].copy()
            dm2 = dm2.loc[dm2["freq"].notna() & dm2["mjd"].notna()].copy()
            if dt2.empty or dm2.empty:
                return merged
            dt2["_freq_bin"] = (dt2["freq"] / float(freq_tol_mhz)).round().astype(int)
            dm2["_freq_bin"] = (dm2["freq"] / float(freq_tol_mhz)).round().astype(int)
            dt2["_orig_idx"] = dt2.index.to_numpy()

            dt2 = dt2.sort_values(["_freq_bin", "mjd"]).reset_index(drop=True)
            dm2 = dm2.sort_values(["_freq_bin", "mjd"]).reset_index(drop=True)

            merged2 = pd.merge_asof(
                dt2,
                dm2,
                on="mjd",
                by="_freq_bin",
                direction="nearest",
                tolerance=float(tol_days),
                suffixes=("", "_meta"),
            )
            merged2 = merged2.set_index("_orig_idx")

            # Fill only rows still missing metadata.
            for col in dm.columns:
                meta_col = f"{col}_meta" if col in dt.columns else col
                if meta_col in merged.columns and meta_col in merged2.columns:
                    merged.loc[mask_unmatched, meta_col] = merged2[meta_col].reindex(
                        merged.loc[mask_unmatched].index
                    )
    return merged
