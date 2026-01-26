"""Backend key normalization helpers.

The pipeline expects every row to have:

- ``sys``: ``TEL.BACKEND.CHANNELFREQ``
- ``group``: ``TEL.BACKEND.BANDFREQ``

TEL/BACKEND/BANDFREQ are primarily parsed from timfile names of the form
``TEL.BACKEND.BANDFREQ.tim``. When ambiguous, the module falls back to flags
or per-row frequency metadata.
"""

from __future__ import annotations
import os
import pandas as pd

KNOWN_TELS = {"EFF", "JBO", "NRT", "WSRT", "SRT", "LEAP"}

def parse_timfile_triplet(timfile_path: str):
    """Parse ``TEL.BACKEND.BANDFREQ`` from a timfile basename.

    Args:
        timfile_path: Path or basename of the timfile.

    Returns:
        Tuple ``(tel, backend, band_mhz)``. Missing values are returned as None.
    """
    base = os.path.basename(str(timfile_path)).strip()
    name = base[:-4] if base.endswith(".tim") else base
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] in KNOWN_TELS:
        tel = parts[0]
        backend = parts[1]
        try:
            band = float(parts[2])
        except ValueError:
            band = None
        return tel, backend, band
    return None, None, None

def ensure_sys_group(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``sys``/``group`` using timfile naming and frequencies.

    Args:
        df: Input DataFrame that may include ``_timfile``, ``freq``, and
            optional ``cenfreq`` metadata columns.

    Returns:
        A copy of ``df`` with ``sys`` and ``group`` columns populated.
    """
    d = df.copy()

    band_from_cen = d.groupby("_timfile")["cenfreq"].median() if "cenfreq" in d.columns else None
    band_from_freq = d.groupby("_timfile")["freq"].median()

    sys_out = []
    group_out = []

    for _, row in d.iterrows():
        sys_val = row.get("sys")
        grp_val = row.get("group")

        if pd.notna(sys_val) and pd.notna(grp_val):
            sys_out.append(str(sys_val))
            group_out.append(str(grp_val))
            continue

        tel, backend, band = parse_timfile_triplet(row.get("_timfile", ""))

        if tel is None:
            for col in ("sys", "group"):
                v = row.get(col)
                if pd.notna(v) and "." in str(v):
                    tel = str(v).split(".")[0]
                    break

        if backend is None:
            for col in ("be", "i", "r"):
                v = row.get(col)
                if pd.notna(v):
                    backend = str(v)
                    break

        tel = tel if tel in KNOWN_TELS else (tel or "UNK")
        backend = (backend or "UNK").upper()

        ch = row.get("freq")
        ch_mhz = int(round(float(ch))) if pd.notna(ch) else None

        band_mhz = None
        if "cenfreq" in d.columns and pd.notna(row.get("cenfreq")):
            band_mhz = int(round(float(row["cenfreq"])))
        elif band is not None:
            band_mhz = int(round(float(band)))
        else:
            tf = row.get("_timfile")
            if band_from_cen is not None and tf in band_from_cen.index and pd.notna(band_from_cen.loc[tf]):
                band_mhz = int(round(float(band_from_cen.loc[tf])))
            elif tf in band_from_freq.index and pd.notna(band_from_freq.loc[tf]):
                band_mhz = int(round(float(band_from_freq.loc[tf])))
            elif ch_mhz is not None:
                band_mhz = ch_mhz

        if pd.isna(sys_val):
            sys_val = f"{tel}.{backend}.{ch_mhz if ch_mhz is not None else 'UNK'}"
        if pd.isna(grp_val):
            grp_val = f"{tel}.{backend}.{band_mhz if band_mhz is not None else 'UNK'}"

        sys_out.append(str(sys_val))
        group_out.append(str(grp_val))

    d["sys"] = sys_out
    d["group"] = group_out
    return d
