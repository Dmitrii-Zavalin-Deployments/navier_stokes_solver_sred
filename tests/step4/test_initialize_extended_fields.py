# tests/step4/test_initialize_extended_fields.py

import numpy as np
from src.step4.initialize_extended_fields import initialize_extended_fields


def test_initialize_extended_fields_basic_copy():
    nx = ny = nz = 2

    U = np.ones((nx+1, ny, nz))
    V = np.ones((nx, ny+1, nz)) * 2
    W = np.ones((nx, ny, nz+1)) * 3

    state = {
        "config": {
            "domain": {"nx": nx, "ny": ny, "nz": nz},
            "initial_conditions": {
                "initial_pressure": 0.0,
                "initial_velocity": [0.0, 0.0, 0.0],
            },
        },
        "mask": np.ones((nx, ny, nz), dtype=int),
        "fields": {"P": np.zeros((nx, ny, nz)), "U": U, "V": V, "W": W},
    }

    out = initialize_extended_fields(state)

    assert np.allclose(out["U_ext"][1:-1, 1:-1, 1:-1], 1.0)
    assert np.allclose(out["V_ext"][0:nx, 1:ny+2, 1:nz+1], 2.0)
    assert np.allclose(out["W_ext"][0:nx, 0:ny, 1:nz+2], 3.0)


def test_initialize_extended_fields_solid_zeroing():
    nx = ny = nz = 2

    mask = np.array([
        [[1, 0],
         [1, 1]]
    ])

    state = {
        "config": {
            "domain": {"nx": nx, "ny": ny, "nz": nz},
            "initial_conditions": {
                "initial_pressure": 5.0,
                "initial_velocity": [7.0, 7.0, 7.0],
            },
        },
        "mask": mask,
        "fields": {
            "P": np.ones((nx, ny, nz)),
            "U": np.ones((nx+1, ny, nz)),
            "V": np.ones((nx, ny+1, nz)),
            "W": np.ones((nx, ny, nz+1)),
        },
    }

    out = initialize_extended_fields(state)

    # Solid cell at (0,0,1) should be zeroed
    assert out["P_ext"][1, 1, 2] == 0.0
