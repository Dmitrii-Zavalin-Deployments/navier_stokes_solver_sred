# tests/step1/test_verify_staggered_shapes.py

import pytest
import numpy as np

from src.step1.verify_staggered_shapes import verify_staggered_shapes


def make_state():
    # Grid definition
    grid = {
        "nx": 2, "ny": 2, "nz": 2,
        "dx": 1.0, "dy": 1.0, "dz": 1.0,
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0,
    }

    # Correct staggered shapes:
    # U: (nx+1, ny, nz) = (3,2,2)
    # V: (nx, ny+1, nz) = (2,3,2)
    # W: (nx, ny, nz+1) = (2,2,3)
    fields = {
        "P": np.zeros((2, 2, 2)).tolist(),
        "U": np.zeros((3, 2, 2)).tolist(),
        "V": np.zeros((2, 3, 2)).tolist(),
        "W": np.zeros((2, 2, 3)).tolist(),
        "Mask": np.zeros((2, 2, 2), dtype=int).tolist(),
    }

    state = {
        "grid": grid,
        "fields": fields,
        "mask_3d": np.zeros((2, 2, 2), dtype=int).tolist(),
        "boundary_table": {},
        "constants": {
            "rho": 1.0, "mu": 0.1, "dt": 0.1,
            "dx": 1.0, "dy": 1.0, "dz": 1.0,
            "inv_dx": 1.0, "inv_dy": 1.0, "inv_dz": 1.0,
            "inv_dx2": 1.0, "inv_dy2": 1.0, "inv_dz2": 1.0,
        },
        "config": {},
    }

    return state


def test_verify_staggered_shapes_failure():
    state = make_state()

    # Break U shape (should be (3,2,2))
    state["fields"]["U"] = np.zeros((2, 2, 2)).tolist()

    with pytest.raises(ValueError):
        verify_staggered_shapes(state)
