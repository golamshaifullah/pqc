import builtins

import numpy as np
import pandas as pd
import pytest

from pqc.features.feature_extraction import (
    add_altaz_features,
    add_orbital_phase,
    add_solar_elongation,
)


def test_add_orbital_phase(tmp_path):
    par = tmp_path / "psr.par"
    par.write_text("PB 2.0\nT0 58000.0\n", encoding="utf-8")
    df = pd.DataFrame({"mjd": [58000.0, 58001.0, 58002.0]})
    out = add_orbital_phase(df, par)
    assert np.allclose(out["orbital_phase"].to_numpy(), [0.0, 0.5, 0.0])


def test_solar_elongation_no_astropy(tmp_path, monkeypatch):
    par = tmp_path / "psr.par"
    par.write_text("ELONG 0.0\nELAT 0.0\n", encoding="utf-8")
    df = pd.DataFrame({"mjd": [58000.0, 58001.0]})

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("astropy"):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    out = add_solar_elongation(df, par)
    assert out["solar_elongation_deg"].isna().all()


def test_solar_elongation_ecliptic_only(tmp_path):
    pytest.importorskip("astropy")

    par = tmp_path / "psr.par"
    par.write_text("ELONG 123.0\nELAT -45.0\n", encoding="utf-8")
    df = pd.DataFrame({"mjd": [58000.0, 58001.0]})

    out = add_solar_elongation(df, par)
    values = out["solar_elongation_deg"].to_numpy()
    assert np.isfinite(values).all()
    assert np.all((0.0 <= values) & (values <= 180.0))


def test_solar_elongation_ecliptic_matches_converted_radec(tmp_path):
    pytest.importorskip("astropy")
    import astropy.units as u
    from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord
    from astropy.time import Time

    df = pd.DataFrame({"mjd": [58000.0, 58010.0, 58100.0]})
    radec_par = tmp_path / "radec.par"
    ecliptic_par = tmp_path / "ecliptic.par"
    radec_par.write_text("RAJ 12:34:56\nDECJ -12:34:56\n", encoding="utf-8")

    psr = SkyCoord("12:34:56", "-12:34:56", unit=(u.hourangle, u.deg), frame="icrs")
    ecliptic = psr.transform_to(BarycentricTrueEcliptic(equinox=Time("J2000")))
    ecliptic_par.write_text(
        f"ELONG {ecliptic.lon.to_value(u.deg):.12f}\n"
        f"ELAT {ecliptic.lat.to_value(u.deg):.12f}\n",
        encoding="utf-8",
    )

    radec_out = add_solar_elongation(df, radec_par)
    ecliptic_out = add_solar_elongation(df, ecliptic_par)
    assert np.allclose(
        radec_out["solar_elongation_deg"].to_numpy(),
        ecliptic_out["solar_elongation_deg"].to_numpy(),
        atol=1e-9,
    )


def test_altaz_no_astropy(tmp_path, monkeypatch):
    par = tmp_path / "psr.par"
    par.write_text("RAJ 00:00:00\nDECJ 00:00:00\n", encoding="utf-8")
    df = pd.DataFrame({"mjd": [58000.0], "tel": ["GBT"]})

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("astropy"):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    out = add_altaz_features(df, par, add_elevation=True, add_airmass=True, add_parallactic=True)
    assert out["elevation_deg"].isna().all()
    assert out["airmass"].isna().all()
    assert out["parallactic_angle_deg"].isna().all()
