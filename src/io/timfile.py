"""Tempo2 timfile parser utilities.

Features:
    - INCLUDE recursion.
    - ``TIME <seconds>`` offsets applied to subsequent TOAs.
    - Ignores comment/control lines (C, #, MODE, FORMAT, EFAC, EQUAD, JUMP).
    - Strict TOA schema: ``filename freq mjd toaerr telescope [flags...]``.

Flag parsing treats negative numeric values (e.g., ``-padd -0.193655``) as
values rather than flag names by distinguishing flag-name tokens from numbers.

Output columns include ``filename``, ``freq``, ``mjd``, ``toaerr_tim``, ``tel``,
``_timfile``, ``_time_offset_sec``, plus any ``-flag`` values as additional
columns.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

SECONDS_PER_DAY = 86400.0
CONTROL_KEYWORDS = {"C", "#", "MODE", "FORMAT", "EFAC", "EQUAD", "JUMP"}

def _is_number(tok: str) -> bool:
    """Return True if ``tok`` parses as a float."""
    try:
        float(tok)
        return True
    except Exception:
        return False

def _is_flag_name(tok: str) -> bool:
    """Return True for flag-name tokens (e.g., ``-sys``), not numeric values."""
    return tok.startswith("-") and not _is_number(tok)

@dataclass
class TimParseResult:
    """Result of parsing timfiles.

    Attributes:
        df: Parsed TOA metadata table.
        dropped_lines: Count of malformed or skipped TOA lines.
    """
    df: pd.DataFrame
    dropped_lines: int

def parse_all_timfiles(all_timfile: str | Path) -> TimParseResult:
    """Parse a tempo2 ``*_all.tim`` file with INCLUDE recursion.

    Args:
        all_timfile: Path to the master timfile.

    Returns:
        Parsed TOA metadata and a count of dropped lines.

    Raises:
        FileNotFoundError: If an INCLUDE target is missing.
        RuntimeError: If no TOAs are parsed from the timfile tree.
    """
    rows = []
    dropped = 0
    all_timfile = Path(all_timfile)

    def parse_timfile_recursive(timfile: Path, time_offset_sec: float) -> float:
        nonlocal dropped
        base_dir = timfile.parent

        with open(timfile) as f:
            for lineno, raw in enumerate(f, start=1):
                stripped = raw.lstrip().strip()
                if not stripped:
                    continue

                upper = stripped.upper()

                if any(upper.startswith(k) for k in CONTROL_KEYWORDS):
                    continue

                if upper.startswith("TIME"):
                    parts = stripped.split()
                    if len(parts) < 2 or not _is_number(parts[1]):
                        dropped += 1
                        continue
                    time_offset_sec = float(parts[1])
                    continue

                if upper.startswith("INCLUDE"):
                    parts = stripped.split()
                    if len(parts) < 2:
                        dropped += 1
                        continue
                    inc = parts[1].strip()
                    inc_path = (base_dir / inc).resolve()
                    if not inc_path.exists():
                        raise FileNotFoundError(str(inc_path))
                    time_offset_sec = parse_timfile_recursive(inc_path, time_offset_sec)
                    continue

                # TOA: filename freq mjd toaerr tel [flags...]
                parts = stripped.split()
                if len(parts) < 5:
                    dropped += 1
                    continue
                if not (_is_number(parts[1]) and _is_number(parts[2]) and _is_number(parts[3])):
                    dropped += 1
                    continue

                row = {
                    "filename": parts[0],
                    "freq": float(parts[1]),
                    "mjd": float(parts[2]) + time_offset_sec / SECONDS_PER_DAY,
                    "toaerr_tim": float(parts[3]),
                    "tel": parts[4],
                    "_timfile": str(timfile),
                    "_time_offset_sec": float(time_offset_sec),
                }

                i = 5
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
    return TimParseResult(df=df, dropped_lines=dropped)
