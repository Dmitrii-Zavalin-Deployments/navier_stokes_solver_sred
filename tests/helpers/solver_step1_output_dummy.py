# tests/helpers/solver_step1_output_dummy.py

import numpy as np
from src.solver_state import SolverState


def make_step1_output_dummy(nx=4, ny=4, nz=4):
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
      - is_fluid (derived from mask)
      - is_boundary_cell (all False)

    Step 1 does NOT fill:
      - operators
      - ppe
      - history  (removed — SolverState no longer accepts it)
    """

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    grid = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
    }

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = {
        "dt": 0.1,
        "external_forces": {},
    }

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    constants = {
        "rho": 1.0,
        "mu": 1.0,
        "dt": config["dt"],
        "dx": grid["dx"],
        "dy": grid["dy"],
        "dz": grid["dz"],
    }

    # ------------------------------------------------------------------
    # Mask (all fluid)
    # ------------------------------------------------------------------
    mask = np.ones((nx, ny, nz), dtype=int)
    is_fluid = mask == 1
    is_boundary_cell = np.zeros_like(mask, dtype=bool)

    # ------------------------------------------------------------------
    # Fields (staggered)
    # ------------------------------------------------------------------
    fields = {
        "P": np.zeros((nx, ny, nz)),
        "U": np.zeros((nx + 1, ny, nz)),
        "V": np.zeros((nx, ny + 1, nz)),
        "W": np.zeros((nx, ny, nz + 1)),
    }

    # ------------------------------------------------------------------
    # Boundary conditions (Step 1 sets this to None or {})
    # ------------------------------------------------------------------
    boundary_conditions = None

    # ------------------------------------------------------------------
    # Empty structures Step 1 does NOT fill
    # ------------------------------------------------------------------
    operators = {}
    ppe = {}
    health = {}

    # ------------------------------------------------------------------
    # Construct SolverState
    # ------------------------------------------------------------------
    return SolverState(
        config=config,
        grid=grid,
        fields=fields,
        mask=mask,
        is_fluid=is_fluid,
        is_boundary_cell=is_boundary_cell,
        constants=constants,
        boundary_conditions=boundary_conditions,
        operators=operators,
        ppe=ppe,
        health=health,
    )
