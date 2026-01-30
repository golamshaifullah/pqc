"""Normalize backend keys for QC grouping.

The pipeline expects every row to have:

- ``sys``: ``TEL.BACKEND.CHANNELFREQ``
- ``group``: ``TEL.BACKEND.BANDFREQ``

TEL/BACKEND/BANDFREQ are primarily parsed from timfile names of the form
``TEL.BACKEND.BANDFREQ.tim``. When ambiguous, the module falls back to flags
or per-row frequency metadata.

See Also:
    pqc.pipeline.run_pipeline: Uses these helpers to normalize metadata.
"""

from __future__ import annotations
import os
import pandas as pd

KNOWN_TELS: set[str] = {"EFF", "JBO", "NRT", "WSRT", "SRT", "LEAP"}
"""Known telescope identifiers used for timfile-derived keys."""

def parse_timfile_triplet(timfile_path: str) -> tuple[str | None, str | None, float | None]:
    """Parse ``TEL[.BACKEND][.BANDFREQ]`` from a timfile basename.

    Args:
        timfile_path (str): Path or basename of the timfile.

    Returns:
        tuple[str | None, str | None, float | None]: Parsed ``(tel, backend,
        band_mhz)`` values. Missing values are returned as None.

    Examples:
        >>> parse_timfile_triplet("EFF.PX.1400.tim")[:2]
        ('EFF', 'PX')
    """
    base = os.path.basename(str(timfile_path)).strip()
    if base.lower().endswith("_all.tim"):
        return None, None, None
    name = base[:-4] if base.endswith(".tim") else base
    parts = name.split(".")
    if len(parts) >= 1:
        tel = parts[0].upper()
        if tel in KNOWN_TELS:
            backend = None
            band = None
            if len(parts) >= 3:
                backend = parts[1]
                band_part = parts[2]
                try:
                    band = float(band_part)
                except ValueError:
                    band = None
            elif len(parts) == 2:
                part = parts[1]
                try:
                    band = float(part)
                except ValueError:
                    backend = part
                    band = None
            return tel, backend, band
    return None, None, None

def ensure_sys_group(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``sys``/``group`` using timfile naming and frequencies.

    Args:
        df (pandas.DataFrame): Input DataFrame that may include ``_timfile``,
            ``freq``, and optional ``cenfreq`` metadata columns.

    Returns:
        pandas.DataFrame: Copy of ``df`` with ``sys`` and ``group`` columns
        populated.

    Notes:
        If a telescope or backend cannot be inferred, ``UNK`` is used. The
        ``group`` key uses a band frequency heuristic based on ``cenfreq``,
        timfile-embedded band, or the per-TOA channel frequency.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"_timfile": ["EFF.PX.1400.tim"], "freq": [1400.0]})
        >>> out = ensure_sys_group(df)
        >>> set(["sys", "group"]).issubset(out.columns)
        True
    """
    d = df.copy()

    band_from_cen = d.groupby("_timfile")["cenfreq"].median() if "cenfreq" in d.columns else None
    band_from_freq = d.groupby("_timfile")["freq"].median()

    sys_out = []
    group_out = []

    def _is_placeholder(val: object) -> bool:
        if pd.isna(val):
            return True
        return str(val).upper().startswith("UNK")

    for _, row in d.iterrows():
        sys_val = row.get("sys")
        grp_val = row.get("group")

        sys_missing = _is_placeholder(sys_val)
        grp_missing = _is_placeholder(grp_val)
        if (not sys_missing) and (not grp_missing):
            sys_out.append(str(sys_val))
            group_out.append(str(grp_val))
            continue

        timfile_guess = row.get("_timfile_base") or row.get("_timfile", "")
        tel, backend, band = parse_timfile_triplet(timfile_guess)
        if tel is None:
            fname = row.get("filename")
            if isinstance(fname, str) and fname.lower().endswith(".tim"):
                tel, backend, band = parse_timfile_triplet(fname)

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

        if sys_missing:
            sys_val = f"{tel}.{backend}.{ch_mhz if ch_mhz is not None else 'UNK'}"
        if grp_missing:
            grp_val = f"{tel}.{backend}.{band_mhz if band_mhz is not None else 'UNK'}"

        sys_out.append(str(sys_val))
        group_out.append(str(grp_val))

    d["sys"] = sys_out
    d["group"] = group_out
    return d
