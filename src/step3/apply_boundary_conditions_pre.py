# src/step3/apply_boundary_conditions_pre.py

import numpy as np

def apply_boundary_conditions_pre(state):
    """
    Apply velocity and pressure boundary conditions before prediction.

    Rules enforced:
      • Zero-out velocities on faces adjacent to solid cells (OR logic).
      • Do NOT zero anything when all cells are fluid.
      • Call BC_handler.apply_pre(state) if provided.
    """

    mask = state["Mask"]
    is_solid = (mask == 0)

    U, V, W = state["U"], state["V"], state["W"]
    nx, ny, nz = mask.shape

    # ----------------------------------------------------------------------
    # Zero faces adjacent to solids (OR logic)
    # ----------------------------------------------------------------------

    # U faces: between i-1 and i
    solid_u = np.zeros_like(U, dtype=bool)
    solid_u[1:-1, :, :] = is_solid[:-1, :, :] | is_solid[1:, :, :]
    U[solid_u] = 0.0

    # V faces: between j-1 and j
    solid_v = np.zeros_like(V, dtype=bool)
    solid_v[:, 1:-1, :] = is_solid[:, :-1, :] | is_solid[:, 1:, :]
    V[solid_v] = 0.0

    # W faces: between k-1 and k
    solid_w = np.zeros_like(W, dtype=bool)
    solid_w[:, :, 1:-1] = is_solid[:, :, :-1] | is_solid[:, :, 1:]
    W[solid_w] = 0.0

    # ----------------------------------------------------------------------
    # Optional boundary condition handler
    # ----------------------------------------------------------------------
    handler = state.get("BC_handler", None)
    if handler and hasattr(handler, "apply_pre"):
        handler.apply_pre(state)
