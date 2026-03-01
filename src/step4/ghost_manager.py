# src/step4/ghost_manager.py

import numpy as np
from src.solver_state import SolverState

def initialize_ghost_fields(state: SolverState) -> None:
    """
    Point 2: Allocate Extended Fields with Ghost Layers.
    Rule 1: O(N^3) memory scaling preserved.
    """
    nx, ny, nz = state.grid.nx, state.grid.ny, state.grid.nz

    # 1. Allocate Extended Fields (nx+2 for centers, nx+3 for staggered faces)
    state.fields.P_ext = np.zeros((nx + 2, ny + 2, nz + 2))
    state.fields.U_ext = np.zeros((nx + 3, ny + 2, nz + 2))
    state.fields.V_ext = np.zeros((nx + 2, ny + 3, nz + 2))
    state.fields.W_ext = np.zeros((nx + 2, ny + 2, nz + 3))

    # 2. Copy Interior (Point 3: Data Migration)
    state.fields.P_ext[1:-1, 1:-1, 1:-1] = state.fields.P
    state.fields.U_ext[1:-1, 1:-1, 1:-1] = state.fields.U
    state.fields.V_ext[1:-1, 1:-1, 1:-1] = state.fields.V
    state.fields.W_ext[1:-1, 1:-1, 1:-1] = state.fields.W