# tests/step4/test_step4_integration_pipeline.py

import numpy as np
from src.step4.orchestrate_step4 import orchestrate_step4


def test_step4_integration_pipeline():
    nx = ny = nz = 2

    state = {
        "config": {
            "domain": {"nx": nx, "ny": ny, "nz": nz},
            "initial_conditions": {
                "initial_pressure": 1.0,
                "initial_velocity": [2.0, 3.0, 4.0],
            },
            "boundary_conditions": [
                {"type": "inlet", "variable": "u", "direction": "x", "side": "min", "value": 10.0}
            ],
        },
        "mask": np.ones((nx, ny, nz), dtype=int),
        "fields": {
            "P": np.ones((nx, ny, nz)),
            "U": np.ones((nx+1, ny, nz)) * 5.0,
            "V": np.ones((nx, ny+1, nz)) * 6.0,
            "W": np.ones((nx, ny, nz+1)) * 7.0,
        },
        "health": {"post_correction_divergence_norm": 0.0},
    }

    out = orchestrate_step4(state)

    # U_ext interior should reflect Step-3 U
    assert np.allclose(out["U_ext"][1:-1, 1:-1, 1:-1], 5.0)

    # BC applied
    assert np.allclose(out["U_ext"][0, :, :], 10.0)

    # Diagnostics
    assert out["diagnostics"]["post_bc_max_velocity"] >= 7.0
