# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState

def make_step1_output_dummy(nx=4, ny=4, nz=4):
    """
    Step 1 Dummy: The Physical Foundation.
    Populates ONLY the attributes filled during initialization.
    """
    # Initialize the 'Empty Slate'
    state = SolverState()

    # Step 1 Responsibility: Geometry
    state.grid = {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": 1.0/nx, "dy": 1.0/ny, "dz": 1.0/nz,
        "x_min": 0.0, "x_max": 1.0,
        "y_min": 0.0, "y_max": 1.0,
        "z_min": 0.0, "z_max": 1.0
    }

    # Step 1 Responsibility: Constants
    state.constants = {
        "rho": 1.0,
        "mu": 0.001,
        "dt": 0.01
    }

    # Step 1 Responsibility: Basic Staggered Fields
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # Step 1 Responsibility: Basic Masking
    state.mask = np.ones((nx, ny, nz), dtype=int)
    state.is_fluid = (state.mask == 1)
    state.is_solid = ~state.is_fluid
    state.is_boundary_cell = np.zeros((nx, ny, nz), dtype=bool)

    return state