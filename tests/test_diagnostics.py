import numpy as np
import pandas as pd

from pqc.utils.diagnostics import export_structure_table


def test_export_structure_table():
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "structure_orbital_phase_chi2": [2.0, 2.0, 5.0],
            "structure_orbital_phase_dof": [3, 3, 4],
            "structure_orbital_phase_p": [0.1, 0.1, 0.01],
            "structure_orbital_phase_present": [False, False, True],
        }
    )
    out = export_structure_table(df, group_cols=("group",))
    assert set(out["group"]) == {"A", "B"}
    assert set(out["feature"]) == {"orbital_phase"}
    assert np.all(np.isfinite(out["chi2"]))
