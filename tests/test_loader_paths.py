from pathlib import Path

from pqc.io.libstempo_loader import _discover_all_tim


def test_discover_all_tim_handles_uppercase_suffix_and_parent_dirs():
    parfile = Path("/tmp/archive.par/files/J1909.PAR")
    assert _discover_all_tim(parfile) == Path("/tmp/archive.par/files/J1909_all.tim")
