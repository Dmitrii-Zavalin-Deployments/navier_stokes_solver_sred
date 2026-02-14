# tests/step4/test_step4_velocity_preservation.py

import numpy as np
from src.step4.orchestrate_step4 import orchestrate_step4


def test_step4_velocity_preservation():
    nx = ny = nz = 2

    state = {
        "config": {
            "domain": {"nx": nx, "ny": ny, "nz": nz},
            "initial_conditions": {
                "initial_pressure": 0.0,
                "initial_velocity": [0.0, 0.0, 0.0],
            },
            "boundary_conditions": [],
        },
        "mask": np.ones((nx, ny, nz), dtype=int),
        "fields": {
            "P": np.zeros((nx, ny, nz)),
            "U": np.ones((nx+1, ny, nz)) * 1.0,
            "V": np.ones((nx, ny+1, nz)) * 2.0,
            "W": np.ones((nx, ny, nz+1)) * 3.0,
        },
        "health": {"post_correction_divergence_norm": 0.0},
    }

    out = orchestrate_step4(state)

    # Interiors preserved
    assert np.allclose(out["U_ext"][1:-1, 1:-1, 1:-1], 1.0)
    assert np.allclose(out["V_ext"][0:nx, 1:ny+2, 1:nz+1], 2.0)
    assert np.allclose(out["W_ext"][0:nx, 0:ny, 1:nz+2], 3.0)

    # Diagnostics reflect max velocity
    assert out["diagnostics"]["post_bc_max_velocity"] == 3.0
