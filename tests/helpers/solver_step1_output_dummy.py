# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState


def make_step1_dummy_state(
    nx=4,
    ny=4,
    nz=4,
    dx=1.0,
    dy=None,
    dz=None,
    dt=0.1,
    rho=1.0,
):
    """
    Canonical dummy representing the EXACT structure of a real post‑Step‑1 SolverState.

    Step 1 fills:
      - grid
      - config
      - constants
      - mask
      - fields
      - boundary_conditions
      - health (empty)

    Step 1 does NOT fill:
      - is_fluid
      - is_boundary_cell
      - operators
      - PPE structure
      - Step‑2 health diagnostics
    """

    dy = dy if dy is not None else dx
    dz = dz if dz is not None else dx

    state = SolverState()

    # Grid
    state.grid = type(
        "Grid",
        (),
        {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": dx,
            "dy": dy,
            "dz": dz,
        },
    )()

    # Config
    state.config = type("Config", (), {"dt": dt})()

    # Constants
    state.constants = {"rho": rho}

    # Mask (all fluid)
    state.mask = np.ones((nx, ny, nz), dtype=int)

    # Fields (staggered)
    state.fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # Boundary conditions
    state.boundary_conditions = {}

    # Health (empty)
    state.health = {}

    return state
