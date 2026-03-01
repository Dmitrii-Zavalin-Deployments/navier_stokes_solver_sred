# src/step2/advection.py

import numpy as np
from src.solver_state import SolverState

def build_advection_stencils(state: SolverState) -> None:
    """
    Step 2 Logic: Allocate memory for 3D interpolation stencils.
    """
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz
    
    dof_u = (nx + 1) * ny * nz
    dof_v = nx * (ny + 1) * nz
    dof_w = nx * ny * (nz + 1)
    total_vel_dof = dof_u + dof_v + dof_w

    # Memory allocation for 8-point trilinear interpolation
    state.advection.weights = np.zeros((total_vel_dof, 8))
    state.advection.indices = np.zeros((total_vel_dof, 8), dtype=int)