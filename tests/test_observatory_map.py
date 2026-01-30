from pqc.features.feature_extraction import _load_observatory_map


def test_load_observatory_map(tmp_path):
    path = tmp_path / "obs.dat"
    path.write_text("1 2 3 SITE1 s1\n4 5 6 SITE2\n", encoding="utf-8")
    m = _load_observatory_map(path)
    assert m["site1"] == (1.0, 2.0, 3.0)
    assert m["s1"] == (1.0, 2.0, 3.0)
    assert m["site2"] == (4.0, 5.0, 6.0)
