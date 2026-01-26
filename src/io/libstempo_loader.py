"""Load libstempo arrays into a DataFrame.

This module extracts TOA-level arrays from ``libstempo.tempopulsar`` and
returns a tidy :class:`pandas.DataFrame`. It includes observing frequencies
used for each TOA.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import libstempo as lt

def load_libstempo(parfile: str | Path) -> pd.DataFrame:
    """Load timing arrays from a ``.par``/``*_all.tim`` pair using libstempo.

    Args:
        parfile: Path to the pulsar ``.par`` file. A sibling ``*_all.tim`` is
            required and will be discovered by filename convention.

    Returns:
        DataFrame with columns ``mjd``, ``resid``, ``sigma``, ``freq``, and
        ``day`` (integer MJD). If available, ``bat_mjd`` is included.

    Raises:
        FileNotFoundError: If ``parfile`` or the matching ``*_all.tim`` is missing.
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
    sigma = np.asarray(psr.toaerrs, dtype="float64")
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
        except Exception:
            pass

    df = pd.DataFrame(out).sort_values("mjd").reset_index(drop=True)
    return df
