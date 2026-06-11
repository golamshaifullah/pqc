import builtins

import numpy as np
import pandas as pd
import pytest

import pqc.features.feature_extraction as feature_extraction
from pqc.features.feature_extraction import (
    _angle_between_cartesian_deg,
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


def test_solar_elongation_default_tempo2_missing(tmp_path, monkeypatch):
    par = tmp_path / "psr.par"
    par.write_text("ELONG 0.0\nELAT 0.0\n", encoding="utf-8")
    df = pd.DataFrame({"mjd": [58000.0, 58001.0]})

    monkeypatch.setattr(feature_extraction.shutil, "which", lambda _name: None)

    out = add_solar_elongation(df, par)
    assert out["solar_elongation_deg"].isna().all()


def test_solar_elongation_astropy_no_astropy(tmp_path, monkeypatch):
    par = tmp_path / "psr.par"
    par.write_text("ELONG 0.0\nELAT 0.0\n", encoding="utf-8")
    df = pd.DataFrame({"mjd": [58000.0, 58001.0]})

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("astropy"):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    out = add_solar_elongation(df, par, solar_elongation_source="astropy")
    assert out["solar_elongation_deg"].isna().all()


def test_solar_elongation_default_tempo2_general2(tmp_path, monkeypatch):
    par = tmp_path / "psr.par"
    tim = tmp_path / "psr_all.tim"
    par.write_text("RAJ 00:00:00\nDECJ 00:00:00\n", encoding="utf-8")
    tim.write_text("FORMAT 1\n", encoding="utf-8")
    df = pd.DataFrame({"mjd": [58000.0, 58001.0, 58002.0]})

    monkeypatch.setattr(feature_extraction.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, capture_output, text, check):
        assert capture_output is True
        assert text is True
        assert check is False
        assert cmd[:4] == ["tempo2", "-output", "general2", "-s"]
        assert "{solarangle}" in cmd[4]
        assert cmd[-2:] == [str(par), str(tim)]

        class Result:
            returncode = 0
            stdout = (
                "Tempo2 noise\n"
                "PQC_SOLAR\t58002.0\t180.0\n"
                "PQC_SOLAR\t58000.0\t0.0\n"
                "PQC_SOLAR\t58001.0\t45.0\n"
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(feature_extraction.subprocess, "run", fake_run)

    out = add_solar_elongation(df, par)
    assert np.allclose(out["solar_elongation_deg"].to_numpy(), [0.0, 45.0, 180.0])


def test_solar_elongation_vector_convention():
    src = np.array(
        [
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    sun = np.array(
        [
            [2.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(_angle_between_cartesian_deg(src, sun), [0.0, 180.0, 90.0])


def test_solar_elongation_ecliptic_only(tmp_path):
    pytest.importorskip("astropy")

    par = tmp_path / "psr.par"
    par.write_text("ELONG 123.0\nELAT -45.0\n", encoding="utf-8")
    df = pd.DataFrame({"mjd": [58000.0, 58001.0]})

    out = add_solar_elongation(df, par, solar_elongation_source="astropy")
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

    radec_out = add_solar_elongation(df, radec_par, solar_elongation_source="astropy")
    ecliptic_out = add_solar_elongation(df, ecliptic_par, solar_elongation_source="astropy")
    assert np.allclose(
        radec_out["solar_elongation_deg"].to_numpy(),
        ecliptic_out["solar_elongation_deg"].to_numpy(),
        atol=1e-9,
    )


def test_solar_elongation_galactic_matches_converted_radec(tmp_path):
    pytest.importorskip("astropy")
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    df = pd.DataFrame({"mjd": [58000.0, 58010.0, 58100.0]})
    radec_par = tmp_path / "radec.par"
    galactic_par = tmp_path / "galactic.par"
    radec_par.write_text("RAJ 12:34:56\nDECJ -12:34:56\n", encoding="utf-8")

    psr = SkyCoord("12:34:56", "-12:34:56", unit=(u.hourangle, u.deg), frame="icrs")
    galactic = psr.galactic
    galactic_par.write_text(
        f"GLONG {galactic.l.to_value(u.deg):.12f}\n" f"GLAT {galactic.b.to_value(u.deg):.12f}\n",
        encoding="utf-8",
    )

    radec_out = add_solar_elongation(df, radec_par, solar_elongation_source="astropy")
    galactic_out = add_solar_elongation(df, galactic_par, solar_elongation_source="astropy")
    assert np.allclose(
        radec_out["solar_elongation_deg"].to_numpy(),
        galactic_out["solar_elongation_deg"].to_numpy(),
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
