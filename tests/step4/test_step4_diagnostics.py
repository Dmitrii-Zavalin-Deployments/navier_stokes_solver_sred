# tests/step4/test_step4_diagnostics.py

import numpy as np
from src.step4.assemble_diagnostics import assemble_diagnostics


def test_diagnostics_basic():
    mask = np.array([
        [[1, 0],
         [1, 1]]
    ])

    state = {
        "mask": mask,
        "U_ext": np.ones((3, 3, 3)),
        "V_ext": np.zeros((3, 3, 3)),
        "W_ext": np.zeros((3, 3, 3)),
        "health": {"post_correction_divergence_norm": 0.5},
    }

    out = assemble_diagnostics(state)
    diag = out["diagnostics"]

    assert diag["total_fluid_cells"] == 3
    assert diag["post_bc_max_velocity"] == 1.0
    assert diag["post_bc_divergence_norm"] == 0.5
    assert diag["initialized"] is True
    assert diag["bc_violation_count"] == 0
