# tests/step4/test_step4_orchestrator_minimal_grid.py

import numpy as np
from src.step4.orchestrate_step4 import orchestrate_step4


def test_step4_minimal_grid():
    nx = ny = nz = 1

    state = {
        "config": {
            "domain": {"nx": nx, "ny": ny, "nz": nz},
            "initial_conditions": {
                "initial_pressure": 1.0,
                "initial_velocity": [2.0, 3.0, 4.0],
            },
            "boundary_conditions": [],
        },
        "mask": np.ones((nx, ny, nz), dtype=int),
        "fields": {
            "P": np.ones((nx, ny, nz)),
            "U": np.ones((nx+1, ny, nz)),
            "V": np.ones((nx, ny+1, nz)),
            "W": np.ones((nx, ny, nz+1)),
        },
        "health": {"post_correction_divergence_norm": 0.0},
    }

    out = orchestrate_step4(state)

    assert out["U_ext"].shape == (nx+3, ny+2, nz+2)
    assert out["V_ext"].shape == (nx+2, ny+3, nz+2)
    assert out["W_ext"].shape == (nx+2, ny+2, nz+3)
    assert out["P_ext"].shape == (nx+2, ny+2, nz+2)
