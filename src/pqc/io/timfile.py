"""Parse tempo2 timfiles into structured metadata tables.

This module implements a permissive parser for tempo2-style timfiles with
INCLUDE recursion. It handles common control lines and converts TOA records
into a pandas DataFrame suitable for merging with libstempo timing arrays.

Features:
    - INCLUDE recursion.
    - ``TIME <seconds>`` offsets applied to subsequent TOAs.
    - Ignores comment/control lines (C, #, MODE, FORMAT, EFAC, EQUAD, JUMP).
    - Parses TOA schemas that optionally include a leading ``sat`` column.

Flag parsing treats negative numeric values (e.g., ``-padd -0.193655``) as
values rather than flag names by distinguishing flag-name tokens from numbers.

Output columns include ``filename``, ``freq``, ``mjd``, ``sat``,
``toaerr_tim``, ``tel``, ``_timfile``, ``_timfile_base``,
``_time_offset_sec``, plus any ``-flag`` values as additional columns.

See Also:
    pqc.io.libstempo_loader.load_libstempo: Load timing arrays via libstempo.
    pqc.io.merge.merge_time_and_meta: Merge timfile metadata with timing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SECONDS_PER_DAY: float = 86400.0
"""Number of seconds in one day (used for TIME offsets)."""

COMMENT_TOKENS: set[str] = {"C", "#"}
"""Tokens that mark an entire line as a comment."""

CONTROL_KEYWORDS: set[str] = {"MODE", "FORMAT", "EFAC", "EQUAD", "JUMP"}
"""Tempo2 timfile control keywords that are ignored by the parser."""


def _parse_float(tok: str) -> float | None:
    """Parse a float token, tolerating Fortran-style exponents."""
    s = tok.strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        pass
    # Fortran D exponent or stray commas
    s2 = s.replace("D", "E").replace("d", "e").replace(",", "")
    try:
        return float(s2)
    except Exception:
        return None


def _is_number(tok: str) -> bool:
    """Return True if a token parses as a float.

    Args:
        tok (str): Candidate token.

    Returns:
        bool: True if ``tok`` parses as a float.
    """
    return _parse_float(tok) is not None


def _is_flag_name(tok: str) -> bool:
    """Return True for flag-name tokens (e.g., ``-sys``), not numeric values.

    Args:
        tok (str): Candidate token.

    Returns:
        bool: True if ``tok`` is a flag-name token.
    """
    return tok.startswith("-") and not _is_number(tok)


@dataclass
class TimParseResult:
    """Hold the result of parsing timfiles.

    Attributes:
        df (pandas.DataFrame): Parsed TOA metadata table.
        dropped_lines (int): Count of malformed or skipped TOA lines.
        commented_lines (int): Count of comment lines (C or #).
        control_lines (int): Count of control/keyword lines (e.g., MODE/FORMAT).
    """

    df: pd.DataFrame
    dropped_lines: int
    commented_lines: int
    control_lines: int


def parse_all_timfiles(
    all_timfile: str | Path,
    *,
    extra_control_keywords: set[str] | None = None,
) -> TimParseResult:
    """Parse a tempo2 ``*_all.tim`` file with INCLUDE recursion.

    Args:
        all_timfile (str | Path): Path to the master timfile.
        extra_control_keywords (set[str] | None): Additional line-start
            keywords to ignore, merged with :data:`CONTROL_KEYWORDS`.

    Returns:
        TimParseResult: Parsed TOA metadata plus bookkeeping counts.

    Raises:
        FileNotFoundError: If an INCLUDE target is missing.
        RuntimeError: If no TOAs are parsed from the timfile tree.

    Examples:
        >>> res = parse_all_timfiles("/data/J1909-3744_all.tim")  # doctest: +SKIP
        >>> {"mjd", "freq"}.issubset(res.df.columns)  # doctest: +SKIP
        True
    """
    rows = []
    dropped = 0
    commented = 0
    control = 0
    all_timfile = Path(all_timfile)
    control_keywords = CONTROL_KEYWORDS | set(extra_control_keywords or [])

    def parse_timfile_recursive(timfile: Path, time_offset_sec: float) -> float:
        nonlocal dropped, commented, control
        base_dir = timfile.parent

        with open(timfile) as f:
            for _lineno, raw in enumerate(f, start=1):
                stripped = raw.lstrip().strip()
                if not stripped:
                    continue

                upper = stripped.upper()

                parts = stripped.split()
                first = parts[0] if parts else ""

                if first in COMMENT_TOKENS:
                    commented += 1
                    continue

                if first in control_keywords:
                    control += 1
                    continue

                if upper.startswith("TIME"):
                    if len(parts) < 2 or not _is_number(parts[1]):
                        dropped += 1
                        continue
                    time_offset_sec = float(_parse_float(parts[1]))
                    continue

                if upper.startswith("INCLUDE"):
                    if len(parts) < 2:
                        dropped += 1
                        continue
                    inc = parts[1].strip()
                    inc_path = (base_dir / inc).resolve()
                    if not inc_path.exists():
                        raise FileNotFoundError(str(inc_path))
                    # Do not leak TIME offsets from included files back into the parent.
                    _ = parse_timfile_recursive(inc_path, time_offset_sec)
                    continue

                # TOA: filename freq mjd toaerr tel [flags...]
                # OR:  sat filename freq mjd toaerr tel [flags...]
                if len(parts) < 4:
                    dropped += 1
                    continue

                row = None
                i = 0

                if _is_number(parts[0]) and len(parts) >= 5:
                    # sat filename freq mjd toaerr tel [flags...]
                    if _is_number(parts[2]) and _is_number(parts[3]) and _is_number(parts[4]):
                        sat = float(_parse_float(parts[0]))
                        tel_tok = parts[5] if len(parts) >= 6 else ""
                        has_tel = bool(tel_tok) and not _is_flag_name(tel_tok)
                        row = {
                            "sat": sat,
                            "filename": parts[1],
                            "freq": float(_parse_float(parts[2])),
                            "mjd": float(_parse_float(parts[3])) + time_offset_sec / SECONDS_PER_DAY,
                            "toaerr_tim": float(_parse_float(parts[4])),
                            "tel": tel_tok if has_tel else "",
                            "_timfile": str(timfile),
                            "_timfile_base": timfile.name,
                            "_time_offset_sec": float(time_offset_sec),
                        }
                        i = 6 if has_tel else 5

                if row is None:
                    if _is_number(parts[1]) and _is_number(parts[2]) and _is_number(parts[3]):
                        mjd = float(_parse_float(parts[2])) + time_offset_sec / SECONDS_PER_DAY
                        tel_tok = parts[4] if len(parts) >= 5 else ""
                        has_tel = bool(tel_tok) and not _is_flag_name(tel_tok)
                        row = {
                            "sat": mjd,
                            "filename": parts[0],
                            "freq": float(_parse_float(parts[1])),
                            "mjd": mjd,
                            "toaerr_tim": float(_parse_float(parts[3])),
                            "tel": tel_tok if has_tel else "",
                            "_timfile": str(timfile),
                            "_timfile_base": timfile.name,
                            "_time_offset_sec": float(time_offset_sec),
                        }
                        i = 5 if has_tel else 4
                    else:
                        # Fallback: find 3 consecutive numeric tokens (freq, mjd, err)
                        num_idx = [j for j, tok in enumerate(parts) if _is_number(tok)]
                        trip = None
                        for j in num_idx:
                            if j + 2 < len(parts) and _is_number(parts[j + 1]) and _is_number(parts[j + 2]):
                                trip = j
                                break
                        if trip is None or trip == 0:
                            dropped += 1
                            continue
                        fname_idx = trip - 1
                        filename = parts[fname_idx]
                        sat_val = None
                        if fname_idx - 1 >= 0 and _is_number(parts[fname_idx - 1]):
                            sat_val = float(_parse_float(parts[fname_idx - 1]))
                        mjd = float(_parse_float(parts[trip + 1])) + time_offset_sec / SECONDS_PER_DAY
                        tel_tok = parts[trip + 3] if trip + 3 < len(parts) else ""
                        has_tel = bool(tel_tok) and not _is_flag_name(tel_tok)
                        row = {
                            "sat": sat_val if sat_val is not None else mjd,
                            "filename": filename,
                            "freq": float(_parse_float(parts[trip])),
                            "mjd": mjd,
                            "toaerr_tim": float(_parse_float(parts[trip + 2])),
                            "tel": tel_tok if has_tel else "",
                            "_timfile": str(timfile),
                            "_timfile_base": timfile.name,
                            "_time_offset_sec": float(time_offset_sec),
                        }
                        i = (trip + 4) if has_tel else (trip + 3)

                while i < len(parts):
                    tok = parts[i]
                    if _is_flag_name(tok):
                        key = tok[1:]
                        if i + 1 < len(parts) and not _is_flag_name(parts[i + 1]):
                            row[key] = parts[i + 1]
                            i += 2
                        else:
                            row[key] = True
                            i += 1
                    else:
                        i += 1

                rows.append(row)

        return time_offset_sec

    parse_timfile_recursive(all_timfile.resolve(), time_offset_sec=0.0)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No TOAs parsed from {all_timfile}")

    # Coerce known numeric metadata flags to numeric types at the source.
    for col in ("cenfreq", "bw"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("mjd").reset_index(drop=True)
    return TimParseResult(
        df=df,
        dropped_lines=dropped,
        commented_lines=commented,
        control_lines=control,
    )
