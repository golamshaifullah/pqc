"""Load libstempo arrays into a DataFrame.

This module extracts TOA-level arrays from :class:`libstempo.tempopulsar` and
returns a tidy :class:`pandas.DataFrame`. It includes observing frequencies
used for each TOA.

See Also:
    pqc.io.timfile.parse_all_timfiles: Parse timfile metadata.
    pqc.io.merge.merge_time_and_meta: Merge metadata with timing arrays.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import libstempo as lt


def load_libstempo(parfile: str | Path) -> pd.DataFrame:
    """Load timing arrays from a ``.par``/``*_all.tim`` pair using libstempo.

    Args:
        parfile (str | Path): Path to the pulsar ``.par`` file. A sibling
            ``*_all.tim`` is required and will be discovered by filename
            convention.

    Returns:
        pandas.DataFrame: Columns ``mjd``, ``resid``, ``sigma``, ``freq``, and
        ``day`` (integer MJD). If available, ``bat_mjd`` is included.

    Raises:
        FileNotFoundError: If ``parfile`` or the matching ``*_all.tim`` is missing.

    Notes:
        This function requires libstempo to be installed and configured with
        a compatible tempo2 installation. If ``psr.bat`` is unavailable in the
        local build, the ``bat_mjd`` column is omitted.

    Examples:
        >>> df = load_libstempo("/data/J1909-3744.par")  # doctest: +SKIP
        >>> {"mjd", "resid", "sigma", "freq"}.issubset(df.columns)  # doctest: +SKIP
        True
    """
    parfile = Path(parfile)
    if not parfile.exists():
        raise FileNotFoundError(str(parfile))

    all_tim = str(parfile).replace(".par", "_all.tim")
    if not Path(all_tim).exists():
        raise FileNotFoundError(all_tim)

    psr = lt.tempopulsar(parfile=str(parfile), timfile=str(all_tim))

    if hasattr(psr, "stoas"):
        mjd = np.asarray(psr.stoas, dtype="float64")
    else:
        mjd = np.asarray(psr.toas(), dtype="float64")
    resid = np.asarray(psr.residuals(), dtype="float64")
    # libstempo toaerrs are in microseconds; convert to seconds to match resid
    sigma = np.asarray(psr.toaerrs, dtype="float64") * 1e-6
    freq = np.asarray(psr.freqs, dtype="float64")

    out = {
        "mjd": mjd,
        "resid": resid,
        "sigma": sigma,
        "freq": freq,
        "day": np.floor(mjd).astype(int),
    }

    # Optional BAT (depends on build)
    if hasattr(psr, "bat"):
        try:
            out["bat_mjd"] = np.asarray(psr.bat(), dtype="float64")
        except Exception as exc:
            from pqc.utils.logging import warn

            warn(f"Failed to read bat_mjd from libstempo: {exc}")

    df = pd.DataFrame(out).sort_values("mjd").reset_index(drop=True)
    return df
