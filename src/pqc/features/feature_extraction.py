"""Add feature columns for PTA QC diagnostics.

This module provides helpers that compute per-TOA covariates such as orbital
phase, solar elongation, and observatory geometry. These features can be
used for feature-domain structure tests or detrending before detection.

See Also:
    pqc.features.backend_keys.ensure_sys_group: Backend grouping helpers.
    pqc.detect.feature_structure: Feature-domain structure diagnostics.
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from pqc.utils.logging import warn


def _read_par_value(parfile: str | Path, key: str) -> float | None:
    """Return a float value for a key in a parfile, or None if missing.

    Args:
        parfile (str | Path): Path to a pulsar ``.par`` file.
        key (str): Parameter name to look up (e.g., ``PB``).

    Returns:
        float | None: Parsed float value, or None if missing/unparseable.
    """
    pat = re.compile(rf"^\s*{re.escape(key)}\s+([^\s]+)")
    for line in Path(parfile).read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def add_orbital_phase(
    df: pd.DataFrame,
    parfile: str | Path,
    *,
    mjd_col: str = "mjd",
) -> pd.DataFrame:
    """Add orbital phase in [0, 1) using PB and T0 from the parfile.

    Args:
        df (pandas.DataFrame): Input table containing timing rows.
        parfile (str | Path): Pulsar ``.par`` file with ``PB`` and ``T0``.
        mjd_col (str): Name of the MJD column in ``df``.

    Returns:
        pandas.DataFrame: Copy of ``df`` with an ``orbital_phase`` column.

    Notes:
        If ``PB`` or ``T0`` is missing or invalid, ``orbital_phase`` is filled
        with NaNs.

    Examples:
        >>> # doctest: +SKIP
        >>> # df = add_orbital_phase(df, "/path/to/psr.par")
    """
    d = df.copy()
    pb = _read_par_value(parfile, "PB")
    t0 = _read_par_value(parfile, "T0")
    if pb is None or t0 is None or pb <= 0:
        d["orbital_phase"] = np.nan
        return d

    mjd = d[mjd_col].to_numpy(dtype=float)
    phase = np.mod((mjd - t0) / pb, 1.0)
    d["orbital_phase"] = phase
    return d


def add_solar_elongation(
    df: pd.DataFrame,
    parfile: str | Path,
    *,
    mjd_col: str = "mjd",
) -> pd.DataFrame:
    """Add solar elongation (deg) between pulsar and Sun at each MJD.

    Args:
        df (pandas.DataFrame): Input table containing timing rows.
        parfile (str | Path): Pulsar ``.par`` file with RA/DEC.
        mjd_col (str): Name of the MJD column in ``df``.

    Returns:
        pandas.DataFrame: Copy of ``df`` with ``solar_elongation_deg``.

    Notes:
        Requires astropy. If astropy is unavailable or RA/DEC are missing,
        ``solar_elongation_deg`` is filled with NaNs and a warning is issued.

    Examples:
        >>> # doctest: +SKIP
        >>> # df = add_solar_elongation(df, "/path/to/psr.par")
    """
    d = df.copy()
    try:
        from astropy.time import Time
        from astropy.coordinates import SkyCoord, get_sun
        import astropy.units as u
    except Exception:
        warn("Astropy is not available; solar elongation will be NaN.")
        d["solar_elongation_deg"] = np.nan
        return d

    raj, decj = _read_par_radec(parfile)
    if raj is None or decj is None:
        warn("RAJ/DECJ missing in parfile; solar elongation will be NaN.")
        d["solar_elongation_deg"] = np.nan
        return d

    psr = SkyCoord(raj, decj, unit=(u.hourangle, u.deg), frame="icrs")
    t = Time(d[mjd_col].to_numpy(dtype=float), format="mjd", scale="tdb")
    sun = get_sun(t).icrs
    sep = psr.separation(sun).to(u.deg).value
    d["solar_elongation_deg"] = sep
    return d


def add_altaz_features(
    df: pd.DataFrame,
    parfile: str | Path,
    *,
    mjd_col: str = "mjd",
    tel_col: str = "tel",
    observatory_path: str | Path | None = None,
    add_elevation: bool = True,
    add_airmass: bool = True,
    add_parallactic: bool = True,
) -> pd.DataFrame:
    """Add elevation, airmass, and parallactic angle using telescope locations.

    Args:
        df (pandas.DataFrame): Input table containing timing rows.
        parfile (str | Path): Pulsar ``.par`` file with RA/DEC.
        mjd_col (str): Name of the MJD column in ``df``.
        tel_col (str): Name of the telescope/site column in ``df``.
        observatory_path (str | Path | None): Optional path to an observatory
            XYZ file for telescope locations.
        add_elevation (bool): If True, add ``elevation_deg``.
        add_airmass (bool): If True, add ``airmass`` (sec(z) with horizon mask).
        add_parallactic (bool): If True, add ``parallactic_angle_deg``.

    Returns:
        pandas.DataFrame: Copy of ``df`` with requested columns added.

    Notes:
        This function requires astropy. If astropy is unavailable or the
        telescope site is unknown, corresponding values are NaN and a warning
        is emitted.

    Examples:
        >>> # doctest: +SKIP
        >>> # df = add_altaz_features(df, "/path/to/psr.par")
    """
    d = df.copy()
    cols = []
    if add_elevation:
        cols.append("elevation_deg")
    if add_airmass:
        cols.append("airmass")
    if add_parallactic:
        cols.append("parallactic_angle_deg")
    for col in cols:
        if col not in d.columns:
            d[col] = np.nan

    try:
        from astropy.time import Time
        from astropy.coordinates import SkyCoord, EarthLocation, AltAz
        import astropy.units as u
    except Exception:
        warn("Astropy is not available; elevation/airmass/parallactic angle will be NaN.")
        return d

    if tel_col not in d.columns:
        warn(f"{tel_col} column missing; elevation/airmass/parallactic angle will be NaN.")
        return d

    raj, decj = _read_par_radec(parfile)
    if raj is None or decj is None:
        warn("RAJ/DECJ missing in parfile; elevation/airmass/parallactic angle will be NaN.")
        return d

    psr = SkyCoord(raj, decj, unit=(u.hourangle, u.deg), frame="icrs")
    ra = psr.ra
    dec = psr.dec

    obs_map = _load_observatory_map(observatory_path)
    for tel in d[tel_col].dropna().unique():
        mask = d[tel_col] == tel
        if not np.any(mask):
            continue
        loc = None
        key = str(tel).strip().lower()
        if key in obs_map:
            x, y, z = obs_map[key]
            loc = EarthLocation.from_geocentric(x, y, z, unit=u.m)
        else:
            try:
                loc = EarthLocation.of_site(str(tel))
            except Exception:
                warn(
                    f"Unknown telescope site '{tel}'; elevation/airmass/parallactic angle will be NaN for this site."
                )
                continue

        t = Time(d.loc[mask, mjd_col].to_numpy(dtype=float), format="mjd", scale="utc")
        altaz = psr.transform_to(AltAz(obstime=t, location=loc))

        if add_elevation:
            d.loc[mask, "elevation_deg"] = altaz.alt.to(u.deg).value
        if add_airmass:
            alt_deg = altaz.alt.to(u.deg).value
            secz = np.array(altaz.secz, dtype=float)
            if hasattr(altaz.secz, "mask"):
                secz = np.where(altaz.secz.mask, np.nan, secz)
            secz = np.where(alt_deg > 0, secz, np.nan)
            d.loc[mask, "airmass"] = secz
        if add_parallactic:
            lst = t.sidereal_time("apparent", longitude=loc.lon)
            H = (lst - ra).wrap_at(180.0 * u.deg).to(u.rad).value
            phi = loc.lat.to(u.rad).value
            dec_rad = dec.to(u.rad).value
            sinH = np.sin(H)
            cosH = np.cos(H)
            tanphi = np.tan(phi)
            num = sinH
            den = tanphi * np.cos(dec_rad) - np.sin(dec_rad) * cosH
            q = np.arctan2(num, den)
            d.loc[mask, "parallactic_angle_deg"] = np.degrees(q)

    return d


def add_freq_bin(
    df: pd.DataFrame,
    *,
    freq_col: str = "freq",
    nbins: int = 8,
    out_col: str = "freq_bin",
) -> pd.DataFrame:
    """Add a linear frequency-bin index column.

    Args:
        df (pandas.DataFrame): Input table containing timing rows.
        freq_col (str): Name of the frequency column.
        nbins (int): Number of linear bins.
        out_col (str): Output column name.

    Returns:
        pandas.DataFrame: Copy of ``df`` with a ``freq_bin``-like column.

    Notes:
        If the frequency range is degenerate or missing, ``out_col`` is set to
        a constant or NaN.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"freq": [1000.0, 1500.0, 2000.0]})
        >>> add_freq_bin(df, nbins=2)["freq_bin"].nunique() <= 2
        True
    """
    d = df.copy()
    if freq_col not in d.columns:
        d[out_col] = np.nan
        return d
    x = pd.to_numeric(d[freq_col], errors="coerce").to_numpy(dtype=float)
    if not np.any(np.isfinite(x)):
        d[out_col] = np.nan
        return d
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        d[out_col] = 0
        return d
    edges = np.linspace(lo, hi, nbins + 1)
    bin_id = np.digitize(x, edges) - 1
    bin_id = np.clip(bin_id, 0, nbins - 1)
    d[out_col] = bin_id
    return d


def _read_par_radec(parfile: str | Path) -> tuple[str | None, str | None]:
    """Read RA/DEC strings from a parfile.

    Args:
        parfile (str | Path): Path to a pulsar ``.par`` file.

    Returns:
        tuple[str | None, str | None]: ``(RAJ, DECJ)`` or ``(None, None)``.
    """
    text = Path(parfile).read_text(encoding="utf-8").splitlines()
    raj = None
    decj = None
    for line in text:
        stripped = line.strip()
        if stripped.startswith("RAJ") or stripped.startswith("RA"):
            parts = stripped.split()
            if len(parts) > 1:
                raj = parts[1]
        if stripped.startswith("DECJ") or stripped.startswith("DEC"):
            parts = stripped.split()
            if len(parts) > 1:
                decj = parts[1]
    return raj, decj


def add_feature_columns(
    df: pd.DataFrame,
    parfile: str | Path,
    *,
    mjd_col: str = "mjd",
    add_orb_phase: bool = True,
    add_solar: bool = True,
    add_elevation: bool = False,
    add_airmass: bool = False,
    add_parallactic: bool = False,
    add_freq: bool = False,
    freq_bins: int = 8,
    observatory_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add configured feature columns for QC.

    Args:
        df (pandas.DataFrame): Input table containing timing rows.
        parfile (str | Path): Pulsar ``.par`` file.
        mjd_col (str): Name of the MJD column in ``df``.
        add_orb_phase (bool): If True, add orbital phase.
        add_solar (bool): If True, add solar elongation.
        add_elevation (bool): If True, add elevation.
        add_airmass (bool): If True, add airmass.
        add_parallactic (bool): If True, add parallactic angle.
        add_freq (bool): If True, add frequency-bin index.
        freq_bins (int): Number of frequency bins if enabled.
        observatory_path (str | Path | None): Optional observatory XYZ file path.

    Returns:
        pandas.DataFrame: Copy of ``df`` with requested feature columns added.

    Examples:
        >>> # doctest: +SKIP
        >>> # df = add_feature_columns(df, "/path/to/psr.par")
    """
    d = df.copy()
    if add_orb_phase:
        d = add_orbital_phase(d, parfile, mjd_col=mjd_col)
    if add_solar:
        d = add_solar_elongation(d, parfile, mjd_col=mjd_col)
    if add_elevation or add_airmass or add_parallactic:
        d = add_altaz_features(
            d,
            parfile,
            mjd_col=mjd_col,
            observatory_path=observatory_path,
            add_elevation=add_elevation,
            add_airmass=add_airmass,
            add_parallactic=add_parallactic,
        )
    if add_freq:
        d = add_freq_bin(d, nbins=freq_bins)
    return d


def _load_observatory_map(
    observatory_path: str | Path | None,
) -> dict[str, tuple[float, float, float]]:
    """Load an observatory name/code to geocentric XYZ mapping.

    Args:
        observatory_path (str | Path | None): Optional path to an observatory
            XYZ file. If None, a bundled data file is used if present.

    Returns:
        dict[str, tuple[float, float, float]]: Mapping of lowercase site names
        and codes to geocentric XYZ coordinates (meters).
    """
    path = None
    if observatory_path is not None:
        path = Path(observatory_path)
    else:
        path = Path(__file__).resolve().parent / "data" / "observatories.dat"
        if not path.exists():
            return {}
    if not path.exists():
        warn(f"Observatory file not found: {path}")
        return {}

    mapping: dict[str, tuple[float, float, float]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
        except Exception:
            continue
        rest = parts[3:]
        if not rest:
            continue
        name = rest[0]
        code = rest[-1] if len(rest) > 1 else None
        mapping[name.strip().lower()] = (x, y, z)
        if code:
            mapping[code.strip().lower()] = (x, y, z)
    return mapping
