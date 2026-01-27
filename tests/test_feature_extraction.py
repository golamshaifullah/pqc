import builtins
import numpy as np
import pandas as pd

from pqc.features.feature_extraction import add_orbital_phase, add_solar_elongation, add_altaz_features

def test_add_orbital_phase(tmp_path):
    par = tmp_path / "psr.par"
    par.write_text("PB 2.0\nT0 58000.0\n", encoding="utf-8")
    df = pd.DataFrame({"mjd": [58000.0, 58001.0, 58002.0]})
    out = add_orbital_phase(df, par)
    assert np.allclose(out["orbital_phase"].to_numpy(), [0.0, 0.5, 0.0])

def test_solar_elongation_no_astropy(tmp_path, monkeypatch):
    par = tmp_path / "psr.par"
    par.write_text("RAJ 00:00:00\nDECJ 00:00:00\n", encoding="utf-8")
    df = pd.DataFrame({"mjd": [58000.0, 58001.0]})

    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name.startswith("astropy"):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    out = add_solar_elongation(df, par)
    assert out["solar_elongation_deg"].isna().all()

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
